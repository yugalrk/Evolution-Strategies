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
CLEARANCE_DISTANCE = 0.03   # 3% safety margin from obstacles (STRICT)
BORDER_MARGIN = 0.025       # 2.5% margin from canvas edges
HARD_PENALTY = 1e20         # Astronomical penalty for obstacle violations
BORDER_PENALTY = 1e18       # Severe penalty for border violations

"""
HOW THE ALGORITHM LEARNS NOT TO CROSS OBSTACLES:

The key is the FITNESS LANDSCAPE created by penalties:

1. EXPONENTIAL PENALTIES create a "repulsion field" around obstacles
   - Far from obstacle: fitness = path_length (what we want to minimize)
   - Near obstacle: fitness = path_length + small_penalty
   - Touching obstacle: fitness = path_length + MASSIVE_penalty
   
2. The algorithm explores the fitness landscape through MUTATION (random noise)
   - Each generation creates a candidate by adding noise to current best
   - If candidate has LOWER fitness (better) → ACCEPT
   - If candidate has HIGHER fitness (worse) → REJECT
   
3. Over generations, the algorithm learns:
   - Mutations toward obstacles → Rejected (high penalty = high fitness)
   - Mutations away from obstacles → Accepted (lower penalty = lower fitness)
   - This creates a "learned behavior" of avoiding obstacles
   
4. SELF-ADAPTATION adjusts exploration strength:
   - Early: Large sigma (mutation strength) = broad exploration
   - Later: Small sigma = fine-tuning around good solutions
   - Algorithm automatically learns when to explore vs exploit!

This is EXACTLY how ES can train neural networks:
- Waypoints = Neural network weights
- Fitness = Negative loss (or reward in RL)
- Obstacles = Constraints (safety, bias, toxicity)
- Mutation = Parameter noise injection
- Selection = Keeping better weights
"""


# --- 3. CORE FITNESS FUNCTION ---

def calculate_fitness(path_vector, source, destination, obstacles, num_waypoints):
    """
    Calculates the cost/fitness (to MINIMIZE) of a given path vector.
    
    THE FITNESS FUNCTION IS HOW THE ALGORITHM "UNDERSTANDS" OBSTACLES:
    
    Good solutions (valid paths):
        fitness = distance (small number, e.g., 0.5)
    
    Bad solutions (crossing obstacles):
        fitness = HUGE_PENALTY + distance (enormous number, e.g., 1e20)
    
    The algorithm doesn't "understand" obstacles conceptually.
    It simply learns that:
    - Some regions of the search space have LOW fitness (good!)
    - Some regions have ASTRONOMICAL fitness (bad! avoid!)
    
    Through trial-and-error (mutation + selection), it discovers
    the LOW fitness regions automatically.
    """
    
    if len(path_vector) == 0:
        waypoints = np.array([]).reshape(0, 2)
    else:
        waypoints = path_vector.reshape(-1, 2)

    # Build full path including source and destination
    if len(waypoints) == 0:
        full_path_points = np.vstack([source, destination])
    else:
        full_path_points = np.vstack([source, waypoints, destination])
    
    distance_cost = 0
    total_penalty = 0
    violation_count = 0
    
    # --- A. HARD BOUNDARY VIOLATIONS (out of bounds) ---
    x_coords = full_path_points[:, 0]
    y_coords = full_path_points[:, 1]

    if np.any(x_coords < X_MIN) or np.any(x_coords > X_MAX) or \
       np.any(y_coords < Y_MIN) or np.any(y_coords > Y_MAX):
        total_penalty += HARD_PENALTY * 1000
        violation_count += 100
    
    # --- B. BORDER MARGIN VIOLATIONS (too close to edges) ---
    # This teaches the algorithm to stay away from boundaries
    if np.any(x_coords < X_MIN + BORDER_MARGIN) or np.any(x_coords > X_MAX - BORDER_MARGIN) or \
       np.any(y_coords < Y_MIN + BORDER_MARGIN) or np.any(y_coords > Y_MAX - BORDER_MARGIN):
        border_violations = np.sum((x_coords < X_MIN + BORDER_MARGIN) | 
                                   (x_coords > X_MAX - BORDER_MARGIN) |
                                   (y_coords < Y_MIN + BORDER_MARGIN) | 
                                   (y_coords > Y_MAX - BORDER_MARGIN))
        violation_count += border_violations
        total_penalty += BORDER_PENALTY * border_violations
    
    # --- C. Distance Cost (what we actually want to minimize) ---
    segments = full_path_points[1:] - full_path_points[:-1]
    segment_lengths = np.sqrt(np.sum(segments**2, axis=1))
    distance_cost = np.sum(segment_lengths)

    # --- D. OBSTACLE COLLISION PENALTIES ---
    # This is THE KEY to teaching obstacle avoidance!
    
    # Check every line segment of the path
    for i in range(len(full_path_points) - 1):
        p1 = tuple(full_path_points[i])
        p2 = tuple(full_path_points[i+1])
        segment_line = LineString([p1, p2])
        
        for obstacle in obstacles:
            dist_to_obstacle = segment_line.distance(obstacle)
            
            # EXPONENTIAL PENALTY creates a steep "repulsion field"
            if dist_to_obstacle < CLEARANCE_DISTANCE:
                violation_count += 1
                violation_depth = CLEARANCE_DISTANCE - dist_to_obstacle
                
                # The closer you get, the EXPONENTIALLY worse it becomes!
                # This creates a strong gradient pushing solutions away
                penalty_multiplier = np.exp(violation_depth * 100)
                total_penalty += HARD_PENALTY * penalty_multiplier
    
    # Also check individual waypoints (point-to-obstacle distance)
    # Double-checking ensures we don't miss any violations
    for waypoint in full_path_points:
        point = Point(waypoint)
        for obstacle in obstacles:
            dist_to_obstacle = point.distance(obstacle)
            
            if dist_to_obstacle < CLEARANCE_DISTANCE:
                violation_count += 1
                violation_depth = CLEARANCE_DISTANCE - dist_to_obstacle
                penalty_multiplier = np.exp(violation_depth * 100)
                # DOUBLE penalty for point violations (even stricter!)
                total_penalty += HARD_PENALTY * penalty_multiplier * 2

    # --- E. Final Cost Calculation ---
    if total_penalty > 0:
        # Invalid solution: Return astronomical cost
        # (distance * 10000 maintains some gradient information)
        return total_penalty + distance_cost * 10000, distance_cost, violation_count

    # Valid solution: Return just the distance
    total_cost = ALPHA_DISTANCE * distance_cost
    
    return total_cost, distance_cost, violation_count


# --- 4. MUTATION FUNCTION (Self-Adaptive (1+1)-ES) ---

def mutate(parent_path, parent_sigma, dimensions):
    """
    Performs self-adaptive mutation on the parent.
    
    THIS IS THE "LEARNING" MECHANISM:
    
    1. We don't use gradients (no backpropagation)
    2. Instead, we add RANDOM NOISE to the current solution
    3. We test if the noisy version is better
    4. If yes → keep it; if no → discard it
    
    SELF-ADAPTATION means the algorithm learns its own learning rate!
    - Sigma (mutation strength) is part of the genome
    - Good sigma values survive; bad ones die out
    - This is like automatic learning rate scheduling in neural nets
    
    CONNECTION TO LLM TRAINING:
    - OpenAI's ES paper showed this can train neural networks
    - Used for RLHF when gradient signals are noisy
    - Highly parallelizable (test many mutations simultaneously)
    - Works well for non-differentiable objectives
    """
    
    # Learning rate based on problem dimensionality
    LEARNING_RATE = 1.0 / np.sqrt(dimensions)
    
    # --- STEP 1: Mutate the mutation strength itself! ---
    # This is the "self-adaptive" part
    tau_prime = LEARNING_RATE / np.sqrt(2 * dimensions)
    tau = LEARNING_RATE
    
    global_factor = np.exp(tau_prime * np.random.normal(0, 1))
    individual_factors = np.exp(tau * np.random.normal(0, 1, dimensions))
    
    new_sigma = parent_sigma * global_factor * individual_factors
    
    # Prevent sigma from becoming too large (chaos) or too small (stagnation)
    new_sigma = np.clip(new_sigma, 0.001, 0.2)
    
    # --- STEP 2: Mutate the solution using the adapted sigma ---
    noise = new_sigma * np.random.normal(0, 1, dimensions)
    new_path = parent_path + noise
    
    # --- STEP 3: Enforce hard constraints (clip to valid bounds) ---
    # Keep coordinates away from borders
    new_path[::2] = np.clip(new_path[::2], X_MIN + BORDER_MARGIN, X_MAX - BORDER_MARGIN)
    new_path[1::2] = np.clip(new_path[1::2], Y_MIN + BORDER_MARGIN, Y_MAX - BORDER_MARGIN)
    
    return new_path, new_sigma, noise


# --- 5. MAIN ES SOLVER FUNCTION ---

def solve_path(params, socketio=None, sid=None, tasks_dict=None):
    """
    Runs the (1+1) Evolution Strategy algorithm.
    
    THE COMPLETE LEARNING LOOP:
    
    1. INITIALIZE: Start with a straight line guess
    2. LOOP for N generations:
        a. MUTATE: Create offspring by adding noise to parent
        b. EVALUATE: Calculate fitness of offspring
        c. SELECT: If offspring is better, it becomes the new parent
        d. ADAPT: Mutation strength automatically adjusts
    3. RETURN: Best solution found
    
    KEY INSIGHT: No explicit "learning" of obstacle positions!
    The algorithm discovers good solutions through:
    - Random exploration (mutation)
    - Fitness-based selection (survival of the fittest)
    - Self-adaptation (learning how to learn)
    
    This is a form of EVOLUTIONARY COMPUTATION - learning without gradients!
    """
    
    # --- Unpack and validate inputs ---
    raw_source = np.array(params['source'])
    raw_destination = np.array(params['destination'])

    # Ensure source and destination respect border margins
    source = np.array([
        np.clip(raw_source[0], X_MIN + BORDER_MARGIN, X_MAX - BORDER_MARGIN),
        np.clip(raw_source[1], Y_MIN + BORDER_MARGIN, Y_MAX - BORDER_MARGIN)
    ])
    destination = np.array([
        np.clip(raw_destination[0], X_MIN + BORDER_MARGIN, X_MAX - BORDER_MARGIN),
        np.clip(raw_destination[1], Y_MIN + BORDER_MARGIN, Y_MAX - BORDER_MARGIN)
    ])
    
    # Convert obstacles to Shapely Polygon objects for efficient collision detection
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
    initial_sigma = params.get('initial_sigma', 0.05)
    dimensions = num_waypoints * 2

    # --- Handle direct path case (no waypoints) ---
    if dimensions == 0:
        path_points = np.vstack([source, destination])
        cost, length, violations = calculate_fitness(np.array([]), source, destination, obstacles, num_waypoints)
        return {"path": path_points.tolist(), "cost": cost, "length": length, "violations": violations}

    # --- INITIALIZATION ---
    # Start with simple linear interpolation (naive initial guess)
    points = np.linspace(source, destination, num_waypoints + 2)
    parent_path = points[1:-1].flatten()  # Exclude source and destination (fixed)
    parent_sigma = np.full(dimensions, initial_sigma)
    parent_cost, parent_distance, parent_violations = calculate_fitness(
        parent_path, source, destination, obstacles, num_waypoints
    )
    
    # Track statistics for analysis
    improvements = 0
    stagnation_counter = 0
    best_valid_cost = float('inf')
    generation_when_valid = -1
    
    # --- EVOLUTION LOOP (1+1)-ES ---
    for gen in range(1, generations + 1):
        # 🛑 STOP CHECK (Multi-tenant isolation)
        if sid and tasks_dict and not tasks_dict.get(sid, True):
            return {"path": [], "cost": 0, "length": 0, "violations": 0, "status": "stopped"}

        # === THE CORE EVOLUTIONARY CYCLE ===
        
        # STEP 1: Generate offspring via mutation (exploration)
        offspring_path, offspring_sigma, mutation_noise = mutate(parent_path, parent_sigma, dimensions)
        
        # STEP 2: Evaluate offspring fitness
        offspring_cost, offspring_distance, offspring_violations = calculate_fitness(
            offspring_path, source, destination, obstacles, num_waypoints
        )

        # STEP 3: Selection (elitist: keep the best)
        if offspring_cost <= parent_cost:
            # Offspring is better or equal → ACCEPT
            if offspring_cost < parent_cost:
                improvements += 1
                stagnation_counter = 0
                
                # Track when we first achieve a valid (violation-free) path
                if offspring_violations == 0 and generation_when_valid == -1:
                    generation_when_valid = gen
                    print(f"\n✅ First valid path found at generation {gen}!")
                
                # Track best valid solution
                if offspring_violations == 0 and offspring_distance < best_valid_cost:
                    best_valid_cost = offspring_distance
            else:
                stagnation_counter += 1
                
            # Accept the offspring
            parent_path = offspring_path
            parent_sigma = offspring_sigma
            parent_cost = offspring_cost
            parent_distance = offspring_distance
            parent_violations = offspring_violations
        else:
            # Offspring is worse → REJECT
            stagnation_counter += 1

        # --- Real-time Visualization Updates ---
        # Stream the evolutionary process to the browser
        if socketio and (gen % 5 == 0 or gen == generations or gen == 1):
            # Current best path
            waypoints = parent_path.reshape(-1, 2)
            current_path_points = np.vstack([source, waypoints, destination])
            
            # Candidate offspring path (showing mutation in action)
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
                "accepted": bool(offspring_cost <= parent_cost)
            }, to=sid)
            time.sleep(0.008)

    # --- Final Results ---
    waypoints = parent_path.reshape(-1, 2)
    full_path_points = np.vstack([source, waypoints, destination])
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"📊 EVOLUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Improvements: {improvements}")
    print(f"Final Violations: {parent_violations}")
    print(f"First Valid Path: Gen {generation_when_valid if generation_when_valid > 0 else 'Never'}")
    print(f"Best Valid Distance: {best_valid_cost:.4f}" if best_valid_cost < float('inf') else "No valid path found")
    print(f"Final Path Length: {parent_distance:.4f}")
    print(f"{'='*60}\n")
    
    return {
        "path": full_path_points.tolist(),
        "cost": float(parent_cost),
        "length": float(parent_distance),
        "violations": int(parent_violations),
        "improvements": improvements
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EVOLUTION STRATEGY PATHFINDER")
    print("="*70)
    print("\n🧬 Demonstrating ES principles applicable to LLM training:")
    print("   • Gradient-free optimization")
    print("   • Self-adaptive learning rates (sigma)")
    print("   • Constraint handling via fitness shaping")
    print("   • Exploration-exploitation balance")
    print("\n💡 Run 'python app.py' to start the interactive demo")
    print("="*70 + "\n")