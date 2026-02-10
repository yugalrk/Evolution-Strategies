# 🧬 Evolution Strategy Pathfinder

An interactive web application that visualizes Evolution Strategies (ES) finding optimal paths through obstacle-filled environments in real-time.

## 🎯 Overview

This application demonstrates how Evolution Strategies, a type of optimization algorithm inspired by biological evolution, can solve pathfinding problems. Watch as the algorithm evolves waypoints to create an optimal path from source to destination while avoiding obstacles.

## ✨ Features

- **Real-time Visualization**: Watch the path evolve generation by generation
- **Interactive Canvas**: Draw custom obstacles, set source and destination points
- **Live Statistics**: Track generation count, cost, path length, improvements, and mutation rate
- **Adjustable Parameters**: Control waypoints, generations, and mutation rates
- **Responsive Design**: Beautiful gradient UI that works on different screen sizes
- **Safety Zones**: Visual clearance margins around obstacles

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Required Packages

Install the required dependencies:

```bash
pip install Flask Flask-SocketIO numpy shapely
```

### Package Details

- **Flask**: Web framework for the server
- **Flask-SocketIO**: Real-time WebSocket communication
- **NumPy**: Numerical computations and array operations
- **Shapely**: Geometric calculations for collision detection

## 📂 Project Structure

```
evolution-pathfinder/
│
├── app.py                 # Flask server with SocketIO
├── es_pathfinder.py       # Evolution Strategy algorithm
├── index.html             # Interactive web interface
└── README.md              # This file
```

## 🎮 How to Use

### 1. Start the Server

Run the Flask application:

```bash
python app.py
```

You should see:
```
🚀 EVOLUTION STRATEGY PATHFINDER SERVER
============================================================
Open your browser and navigate to:
➡️  http://127.0.0.1:5000/
```

### 2. Open in Browser

Navigate to `http://127.0.0.1:5000/` in your web browser.

### 3. Set Up the Problem

1. **Set Source Point** (Green 🟢):
   - Select "Set Source" from the dropdown
   - Click anywhere on the canvas to place the starting point

2. **Set Destination Point** (Blue 🔵):
   - Select "Set Destination" from the dropdown
   - Click anywhere on the canvas to place the goal point

3. **Draw Obstacles** (Red 🚧):
   - Select "Draw Obstacles" from the dropdown
   - Click and drag to create rectangular obstacles
   - Create multiple obstacles to make the problem more challenging

### 4. Configure Parameters

Adjust the algorithm parameters as needed:

- **Waypoints**: Number of intermediate points (5-50)
  - More waypoints = more flexible paths but slower evolution
  - Fewer waypoints = simpler paths but may not navigate complex obstacles

- **Generations**: Number of evolution iterations (100-5000)
  - More generations = better optimization but longer runtime
  - Start with 1000 for most problems

- **Initial Mutation Rate** (Sigma): Starting mutation strength (0.01-0.5)
  - Higher values = more exploration but less precision
  - Lower values = more exploitation of known good solutions
  - The algorithm adapts this automatically during evolution

### 5. Run Evolution

Click the **"▶ Run Evolution"** button and watch:

- **Yellow dots**: Waypoints being optimized
- **Black line**: Current best path
- **Statistics**: Real-time updates on performance
- **Progress bar**: Evolution completion status

## 🧮 Algorithm Explanation

### Evolution Strategy (1+1)-ES

The application uses a **(1+1) Evolution Strategy** with self-adaptive mutation:

1. **Initialization**: 
   - Waypoints are initialized along a straight line from source to destination

2. **Mutation**:
   - Each generation creates one offspring by mutating the parent path
   - Mutation strength (sigma) adapts automatically using the "1/5 success rule"

3. **Selection**:
   - If the offspring is better (or equal), it replaces the parent
   - Otherwise, the parent survives

4. **Fitness Function**:
   - **Primary**: Minimize path length (Euclidean distance)
   - **Penalty**: Massive cost for violating obstacle clearance
   - **Penalty**: Heavy cost for going out of bounds

### Self-Adaptive Mutation

The algorithm adjusts its own mutation rate (sigma) during evolution:

- **Global component**: Affects all parameters equally
- **Individual components**: Each coordinate can mutate differently
- **Learning rate**: Scales with problem dimensionality

This allows the algorithm to:
- Explore broadly early in evolution
- Refine solutions precisely as it converges

## 📊 Understanding the Statistics

- **Status**: Current algorithm state (Ready, Evolving, Complete)
- **Generation**: Current iteration number
- **Best Cost**: Total fitness (lower is better)
  - Shows "INVALID" if path violates constraints
- **Path Length**: Euclidean distance of the path
- **Improvements**: Number of times a better solution was found
- **Avg Sigma**: Average mutation strength (adapts over time)

## 🎨 Visual Elements

| Color | Element | Description |
|-------|---------|-------------|
| 🟢 Green | Source | Starting point (S) |
| 🔵 Blue | Destination | Goal point (D) |
| 🔴 Red | Obstacles | Forbidden regions |
| 🟡 Yellow | Waypoints | Points being evolved |
| ⚫ Black | Path | Current best solution |
| 🟠 Orange (transparent) | Safety Zone | Minimum clearance margin |

## 💡 Tips for Best Results

1. **Start Simple**: Begin with 2-3 obstacles to understand the behavior
2. **Increase Complexity**: Add more obstacles for challenging mazes
3. **Adjust Waypoints**: Use more waypoints for complex obstacle arrangements
4. **Be Patient**: Complex problems may need 2000+ generations
5. **Watch the Evolution**: Notice how the path gradually improves
6. **Experiment**: Try different parameter combinations

## 🔧 Troubleshooting

### "Disconnected" Error
- Ensure `app.py` is running
- Check that no other service is using port 5000
- Refresh the browser page

### "No Valid Path Found"
- Increase the number of generations
- Reduce the number of waypoints and try again
- Check if a path is actually possible (remove some obstacles)
- Increase initial mutation rate for more exploration

### Slow Performance
- Reduce the number of waypoints
- Reduce the number of generations
- Clear some obstacles

### Path Goes Through Obstacles
- Increase the number of generations
- The red transparent zones show the required clearance
- Check that obstacles aren't too close together

## 🔬 Technical Details

### Coordinate System
- Canvas uses normalized coordinates [0, 1] for portability
- Clearance distance: 0.01 (1% of canvas width)
- All calculations in relative coordinates, converted for display

### Real-time Updates
- WebSocket connection via Socket.IO
- Updates every 10 generations for smooth visualization
- Non-blocking architecture for responsive UI

### Fitness Calculation
```
if (collision or out_of_bounds):
    fitness = HARD_PENALTY + distance
else:
    fitness = path_length
```

## 🎓 Educational Value

This project demonstrates:
- **Evolutionary Algorithms**: Self-adaptive optimization
- **Pathfinding**: Constrained optimization in 2D space
- **Real-time Visualization**: Live algorithm performance
- **Web Technologies**: Flask, SocketIO, Canvas API
- **Computational Geometry**: Collision detection with Shapely

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to fork, modify, and experiment with the code!

## 📧 Support

If you encounter issues or have questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure Python 3.7+ is being used

---

**Enjoy watching evolution find the optimal path! 🧬✨**