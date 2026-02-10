import numpy as np
from shapely.geometry import LineString, Polygon, Point
import time

# --- 1. CONFIGURATION & ENVIRONMENT SETUP ---

# Define the 2D search space (normalized coordinates 0-1)
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0

# --- 2. EVOLUTION STRATEGY PARAMETERS & CONSTRAINTS ---

# Fitness weights and Constraints
ALPHA_DISTANCE = 1.0
CLEARANCE_DISTANCE = 0.02  # Normalized clearance (2% of canvas) - increased for better safety
HARD_PENALTY = 1e15        # Extremely large penalty to strongly discourage violations


# --- 3. CORE FITNESS FUNCTION ---

def calculate_fitness(path_vector, source, destination, obstacles, num_waypoints):
    """
    Calculates the cost/fitness (to MINIMIZE) of a given path vector.
    Applies strict penalties for boundary violations and obstacle clearance violations.
    """
    
    if len(path_vector) == 0:
        # Direct path from source to destination
        waypoints = np.array([]).reshape(0, 2)
    else:
        # Reshape the flat vector into waypoints
        waypoints = path_vector.reshape(-1, 2)

    # Build full path including source and destination
    if len(waypoints) == 0:
        full_path_points = np.vstack([source, destination])
    else:
        full_path_points = np.vstack([source, waypoints, destination])
    
    distance_cost = 0
    total_penalty = 0
    violation_count = 0
    
    # --- A. Boundary Check Penalty ---
    x_coords = full_path_points[:, 0]
    y_coords = full_path_points[:, 1]

    if np.any(x_coords < X_MIN) or np.any(x_coords > X_MAX) or \
       np.any(y_coords < Y_MIN) or np.any(y_coords > Y_MAX):
        total_penalty += HARD_PENALTY * 100
        violation_count += 1
    
    # --- B. Distance Cost (Euclidean length) ---
    segments = full_path_points[1:] - full_path_points[:-1]
    segment_lengths = np.sqrt(np.sum(segments**2, axis=1))
    distance_cost = np.sum(segment_lengths)

    # --- C. Strict Collision & Clearance Penalty ---
    for i in range(len(full_path_points) - 1):
        p1 = tuple(full_path_points[i])
        p2 = tuple(full_path_points[i+1])
        segment_line = LineString([p1, p2])
        
        for obstacle in obstacles:
            dist_to_obstacle = segment_line.distance(obstacle)
            
            # Check if violates clearance
            if dist_to_obstacle < CLEARANCE_DISTANCE:
                violation_count += 1
                violation_depth = CLEARANCE_DISTANCE - dist_to_obstacle
                
                # Exponential penalty based on how deep the violation is
                penalty_multiplier = np.exp(violation_depth * 50)
                total_penalty += HARD_PENALTY * penalty_multiplier
    
    # Also check waypoints themselves (not just segments)
    for waypoint in full_path_points:
        point = Point(waypoint)
        for obstacle in obstacles:
            dist_to_obstacle = point.distance(obstacle)
            if dist_to_obstacle < CLEARANCE_DISTANCE:
                violation_count += 1
                violation_depth = CLEARANCE_DISTANCE - dist_to_obstacle
                penalty_multiplier = np.exp(violation_depth * 50)
                total_penalty += HARD_PENALTY * penalty_multiplier

    # --- D. Final Cost Calculation ---
    if total_penalty > 0:
        # Return extremely high cost for invalid solutions
        return total_penalty + distance_cost * 1000, distance_cost, violation_count

    # If no penalties, cost is the path length
    total_cost = ALPHA_DISTANCE * distance_cost
    
    return total_cost, distance_cost, violation_count


# --- 4. MUTATION FUNCTION (Self-Adaptive (1+1)-ES) ---

def mutate(parent_path, parent_sigma, dimensions):
    """Performs self-adaptive mutation on the parent."""
    
    # Learning rate based on dimensions
    LEARNING_RATE = 1.0 / np.sqrt(dimensions)
    
    # 1. Mutate Strategy Parameters (Sigma)
    tau_prime = LEARNING_RATE / np.sqrt(2 * dimensions)
    global_factor = np.exp(tau_prime * np.random.normal(0, 1))
    individual_factors = np.exp(LEARNING_RATE * np.random.normal(0, 1, dimensions))
    new_sigma = parent_sigma * global_factor * individual_factors
    
    # Clamp sigma to reasonable bounds
    new_sigma = np.clip(new_sigma, 0.001, 0.3)
    
    # 2. Mutate Path Parameters
    noise = new_sigma * np.random.normal(0, 1, dimensions)
    new_path = parent_path + noise
    
    # Boundary enforcement: Clip coordinates to map bounds
    new_path[::2] = np.clip(new_path[::2], X_MIN, X_MAX)    # x-coordinates
    new_path[1::2] = np.clip(new_path[1::2], Y_MIN, Y_MAX)  # y-coordinates
    
    return new_path, new_sigma, noise


# --- 5. MAIN ES SOLVER FUNCTION ---

def solve_path(params, socketio=None):
    """Runs the ES algorithm and streams updates via SocketIO."""
    
    # --- Unpack and validate inputs ---
    raw_source = np.array(params['source'])
    raw_destination = np.array(params['destination'])

    # Ensure source and destination are within bounds
    source = np.array([
        np.clip(raw_source[0], X_MIN, X_MAX),
        np.clip(raw_source[1], Y_MIN, Y_MAX)
    ])
    destination = np.array([
        np.clip(raw_destination[0], X_MIN, X_MAX),
        np.clip(raw_destination[1], Y_MIN, Y_MAX)
    ])
    
    # Convert obstacles to Shapely Polygon objects
    obstacles = [
        Polygon([
            (o['x1'], o['y1']), 
            (o['x2'], o['y1']), 
            (o['x2'], o['y2']), 
            (o['x1'], o['y2'])
        ]) 
        for o in params['obstacles']
    ]
    
    num_waypoints = params.get('num_waypoints', 20)
    generations = params.get('generations', 1000)
    initial_sigma = params.get('initial_sigma', 0.1)
    dimensions = num_waypoints * 2

    # --- Handle direct path case ---
    if dimensions == 0:
        path_points = np.vstack([source, destination])
        cost, length, violations = calculate_fitness(np.array([]), source, destination, obstacles, num_waypoints)
        return {"path": path_points.tolist(), "cost": cost, "length": length, "violations": violations}

    # --- INITIALIZATION ---
    # Start with linear interpolation from source to destination
    points = np.linspace(source, destination, num_waypoints + 2)
    parent_path = points[1:-1].flatten()  # Exclude source and destination
    parent_sigma = np.full(dimensions, initial_sigma)
    parent_cost, parent_distance, parent_violations = calculate_fitness(
        parent_path, source, destination, obstacles, num_waypoints
    )
    
    # Track statistics
    improvements = 0
    stagnation_counter = 0
    best_valid_cost = float('inf')
    
    # --- EVOLUTION LOOP ---
    for gen in range(1, generations + 1):
        # Generate offspring with noise
        offspring_path, offspring_sigma, mutation_noise = mutate(parent_path, parent_sigma, dimensions)
        offspring_cost, offspring_distance, offspring_violations = calculate_fitness(
            offspring_path, source, destination, obstacles, num_waypoints
        )

        # Selection: (1+1)-ES - offspring replaces parent if better or equal
        if offspring_cost <= parent_cost:
            if offspring_cost < parent_cost:
                improvements += 1
                stagnation_counter = 0
                
                # Track best valid solution
                if offspring_violations == 0 and offspring_distance < best_valid_cost:
                    best_valid_cost = offspring_distance
            else:
                stagnation_counter += 1
                
            parent_path = offspring_path
            parent_sigma = offspring_sigma
            parent_cost = offspring_cost
            parent_distance = offspring_distance
            parent_violations = offspring_violations

        else:
            stagnation_counter += 1

        # --- Emit real-time updates via WebSocket ---
        # Show updates more frequently, and include the candidate offspring for visualization
        if socketio and (gen % 5 == 0 or gen == generations or gen == 1):
            # Current best path
            waypoints = parent_path.reshape(-1, 2)
            current_path_points = np.vstack([source, waypoints, destination])
            
            # Candidate offspring path (for visualization)
            offspring_waypoints = offspring_path.reshape(-1, 2)
            candidate_path_points = np.vstack([source, offspring_waypoints, destination])
            
            socketio.emit('path_update', {
                "path": current_path_points.tolist(),
                "candidate_path": candidate_path_points.tolist(),
                "generation": gen,
                "cost": float(parent_cost),
                "length": float(parent_distance),
                "violations": int(parent_violations),
                "improvements": improvements,
                "avg_sigma": float(np.mean(parent_sigma)),
                "accepted": bool(offspring_cost <= parent_cost)  # Was the candidate accepted?
            })
            time.sleep(0.008)  # Small delay for visualization

    # --- PREPARE FINAL RESULTS ---
    waypoints = parent_path.reshape(-1, 2)
    full_path_points = np.vstack([source, waypoints, destination])
    
    return {
        "path": full_path_points.tolist(),
        "cost": float(parent_cost),
        "length": float(parent_distance),
        "violations": int(parent_violations),
        "improvements": improvements
    }


# This section is for local testing only
if __name__ == "__main__":
    print("This file contains the ES logic. Run 'app.py' to start the web server.")