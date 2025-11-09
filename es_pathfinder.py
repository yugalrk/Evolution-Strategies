import numpy as np
from shapely.geometry import LineString, Polygon, Point
import time
import math

# --- 1. CONFIGURATION & ENVIRONMENT SETUP ---

# Define the 2D search space (Must match Canvas size in HTML)
X_MIN, X_MAX = 0.0, 1
Y_MIN, Y_MAX = 0.0, 1

# --- 2. EVOLUTION STRATEGY PARAMETERS & CONSTRAINTS ---

# Fitness weights and Constraints
ALPHA_DISTANCE = 1.0
CLEARANCE_DISTANCE = 0.01 # Required minimum distance (safety margin) from obstacles
HARD_PENALTY = 1e9       # Massive penalty to forbid clearance violations


# --- 3. CORE FITNESS FUNCTION ---

def calculate_fitness(path_vector, source, destination, obstacles, num_waypoints):
    """
    Calculates the cost/fitness (to MINIMIZE) of a given path vector.
    Applies penalties for boundary violations and obstacle clearance violations.
    """
    
    # Reshape the flat vector into a list of waypoints (x, y)
    waypoints = path_vector.reshape(-1, 2)

    # 1. Total Path Segments (including Source and Destination)
    full_path_points = np.vstack([source, waypoints, destination])
    distance_cost = 0
    total_penalty = 0
    
    # --- A. Boundary Check Penalty (Ensures path stays within 0-600) ---
    x_coords = full_path_points[:, 0]
    y_coords = full_path_points[:, 1]

    # This check acts as a final safeguard, though the clip in solve_path and mutate should handle it.
    if np.any(x_coords < X_MIN) or np.any(x_coords > X_MAX) or \
       np.any(y_coords < Y_MIN) or np.any(y_coords > Y_MAX):
        # Apply a massive penalty if any point is out of bounds
        total_penalty += HARD_PENALTY * 10
    
    # --- B. Distance Cost (Euclidean length) ---
    segments = full_path_points[1:] - full_path_points[:-1]
    segment_lengths = np.sqrt(np.sum(segments**2, axis=1))
    distance_cost = np.sum(segment_lengths)

    # --- C. Collision & Clearance Penalty ---
    
    # Iterate through all path segments and check for clearance violation
    for i in range(len(full_path_points) - 1):
        # Using tuples for Shapely LineString creation
        p1 = tuple(full_path_points[i])
        p2 = tuple(full_path_points[i+1])
        segment_line = LineString([p1, p2])
        
        for obstacle in obstacles:
            dist_to_obstacle = segment_line.distance(obstacle)
            
            if dist_to_obstacle < CLEARANCE_DISTANCE:
                # Apply penalty based on violation depth for gradient
                violation_depth = CLEARANCE_DISTANCE - dist_to_obstacle
                total_penalty += HARD_PENALTY + (violation_depth * 1000)

    # --- D. Final Cost Calculation ---
    if total_penalty > 0:
        return total_penalty + distance_cost, distance_cost

    # If no penalties, the cost is simply the path's length.
    total_cost = ALPHA_DISTANCE * distance_cost
    
    return total_cost, distance_cost

# --- 4. MUTATION FUNCTION (Self-Adaptive (1+1)-ES) ---

def mutate(parent_path, parent_sigma, dimensions):
    """Performs self-adaptive mutation on the parent."""
    
    # Calculate Learning Rate dynamically based on dimensions
    LEARNING_RATE = 1.0 / np.sqrt(dimensions)
    
    # 1. Mutate Strategy Parameters (Sigma)
    tau_prime = LEARNING_RATE / np.sqrt(2 * dimensions)
    global_factor = np.exp(tau_prime * np.random.normal(0, 1))
    individual_factors = np.exp(LEARNING_RATE * np.random.normal(0, 1, dimensions))
    new_sigma = parent_sigma * global_factor * individual_factors
    
    # 2. Mutate Path Parameters (P)
    noise = new_sigma * np.random.normal(0, 1, dimensions)
    new_path = parent_path + noise
    
    # Boundary Check: Clip the coordinates to stay within the map bounds
    new_path[::2] = np.clip(new_path[::2], X_MIN, X_MAX) # x-coordinates
    new_path[1::2] = np.clip(new_path[1::2], Y_MIN, Y_MAX) # y-coordinates
    
    return new_path, new_sigma

# --- 5. MAIN ES SOLVER FUNCTION ---

def solve_path(params, socketio=None):
    """Runs the ES algorithm and streams updates via SocketIO."""
    
    # --- Unpack and CLIP Initial Points (Crucial Fix) ---
    raw_source = np.array(params['source'])
    raw_destination = np.array(params['destination'])

    # Ensure S and D are strictly within bounds before starting the search
    source = np.array([
        np.clip(raw_source[0], X_MIN, X_MAX),
        np.clip(raw_source[1], Y_MIN, Y_MAX)
    ])
    destination = np.array([
        np.clip(raw_destination[0], X_MIN, X_MAX),
        np.clip(raw_destination[1], Y_MIN, Y_MAX)
    ])
    
    # Convert obstacle dictionary format (x1, y1, x2, y2) to Shapely Polygon objects
    obstacles = [
        Polygon([(o['x1'], o['y1']), (o['x2'], o['y1']), (o['x2'], o['y2']), (o['x1'], o['y2'])]) 
        for o in params['obstacles']
    ]
    
    num_waypoints = params.get('num_waypoints', 20)
    generations = params.get('generations', 1000)
    initial_sigma = params.get('initial_sigma', 1.0)
    dimensions = num_waypoints * 2

    if dimensions == 0:
        path_points = np.vstack([source, destination])
        cost, length = calculate_fitness(np.array([]), source, destination, obstacles, num_waypoints)
        return {"path": path_points.tolist(), "cost": cost, "length": length}

    # --- INITIALIZATION ---
    points = np.linspace(source, destination, num_waypoints + 2)
    parent_path = points[1:-1].flatten()
    parent_sigma = np.full(dimensions, initial_sigma)
    parent_cost, parent_distance = calculate_fitness(parent_path, source, destination, obstacles, num_waypoints)
    
    # --- EVOLUTION LOOP ---
    for gen in range(1, generations + 1):
        offspring_path, offspring_sigma = mutate(parent_path, parent_sigma, dimensions)
        offspring_cost, offspring_distance = calculate_fitness(offspring_path, source, destination, obstacles, num_waypoints)

        # Selection (1+1)-ES: Offspring replaces parent if it's better (lower cost)
        if offspring_cost <= parent_cost:
            parent_path = offspring_path
            parent_sigma = offspring_sigma
            parent_cost = offspring_cost
            parent_distance = offspring_distance

        # --- Emit real-time updates via WebSocket ---
        if socketio and (gen % 25 == 0 or gen == generations):
            waypoints = parent_path.reshape(-1, 2)
            current_path_points = np.vstack([source, waypoints, destination])
            socketio.emit('path_update', {
                "path": current_path_points.tolist(),
                "generation": gen,
                "cost": float(parent_cost),
                "length": float(parent_distance),
            })
            time.sleep(0.01) # Yield to allow server to process

    # --- PREPARE FINAL RESULTS ---
    waypoints = parent_path.reshape(-1, 2)
    full_path_points = np.vstack([source, waypoints, destination])
    
    return {
        "path": full_path_points.tolist(),
        "cost": float(parent_cost),
        "length": float(parent_distance),
    }

# This section is for local testing only
if __name__ == "__main__":
    print("This file contains the ES logic. Run 'app.py' to start the web server.")