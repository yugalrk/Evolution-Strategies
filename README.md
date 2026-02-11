# 🧬 Evolution Strategy Pathfinder (Production Edition)

An interactive, high-performance web application that visualizes **Evolution Strategies (ES)** finding optimal paths through obstacle-filled environments in real-time. This version is optimized for cloud deployment (Render) and supports multiple concurrent users.

## 🎯 Overview

This application demonstrates how Evolution Strategies—a gradient-free optimization technique inspired by biological evolution—can solve complex pathfinding problems. It is a direct visualization of the same principles used in **RLHF (Reinforcement Learning from Human Feedback)** to train Large Language Models (LLMs) to avoid "obstacles" like toxicity and bias.

## ✨ New & Advanced Features

* **Multi-Tenant Isolation**: Uses Socket.IO Rooms (`sid`) to allow multiple users to run independent simulations simultaneously without cross-talk.
* **Mobile-Optimized UX**:
* **Tap-to-Place**: A two-tap system for creating obstacles, perfect for touchscreens.
* **Crosshair Helper**: Visual alignment guides for precise placement on mobile.


* **Quick-Load Presets**: Four pre-built challenge maps (**Simple, Maze, The Wall, The Trap**) to see the AI in action instantly.
* **Emergency Stop/Reset**: A server-side interrupt signal that kills the Python loop immediately when the user resets the map.
* **Self-Adaptive Mutation**: Watch the ** (Sigma)** value evolve in real-time as the AI learns how to learn.

## 🚀 Installation & Deployment

### Required Packages

```bash
pip install Flask Flask-SocketIO eventlet gunicorn numpy shapely

```

### Production Deployment (Render/Heroku)

This project is configured for **Gunicorn** using the **Eventlet** worker class to handle real-time WebSocket concurrency.

**Start Command:**

```bash
gunicorn -k eventlet -w 1 --timeout 120 app.py:app

```

## 📂 Project Structure

```
evolution-pathfinder/
│
├── app.py              # Flask + SocketIO (Multi-tenant & Session Logic)
├── es_pathfinder.py    # The (1+1)-ES Engine (Gradient-free Optimization)
├── index.html          # Frontend (Canvas API + Pointer Events)
└── requirements.txt    # Production dependencies

```

## 🎮 How to Use

### 1. Quick Start (Recommended)

Select one of the **Quick Load Maps**:

* **🌀 Maze**: Complex navigation through narrow gaps.
* **🕳️ The Trap**: A "C-shape" designed to test if the AI can escape local optima.

### 2. Manual Setup

1. **Set Points**: Use the dropdown to place your **Source (🟢)** and **Destination (🔵)**.
2. **Draw Obstacles**:
* **Desktop**: Click and drag to draw red forbidden zones.
* **Mobile**: Tap once for the first corner, tap again for the opposite corner.


3. **Adjust Parameters**:
* **Waypoints**: The "DNA" of your path. More points = more flexibility.
* **Mutation Rate ()**: How much "risk" the AI takes in its guesses.



### 3. Watch the Evolution

Click **"▶ Run Evolution"**:

* **Purple Dashed Lines**: The "Offspring" (mutations) being tested.
* **Black Line & Yellow Dots**: The "Parent" (current champion).
* **Avg Sigma**: Watch as the AI automatically lowers its mutation rate to fine-tune the final path.

## 🧮 The Science: Why ES?

Unlike standard Neural Network training which uses **Backpropagation** (calculating gradients), ES uses **Random Perturbation**:

1. **Mutation**: Add random noise to the current best path.
2. **Evaluation**: Check the path against the **Fitness Function** (Length + Obstacle Penalties).
3. **Selection**: If the noisy path is better, it becomes the new standard.

**Why this matters for LLMs:** When we train AI on human values (Safety/Ethics), we don't always have a mathematical gradient for "common sense." ES allows us to "evolve" the model toward safer outputs by penalizing "toxic obstacles."

## 📊 Technical Features

* **Asynchronous Loop**: The server runs the evolution in a separate thread, emitting updates every 5 generations to keep the UI fluid.
* **Coordinate Normalization**: All math is performed on a `[0, 1]` scale, ensuring the simulation behaves identically on a 4K monitor or a smartphone.
* **Shapely Geometry**: Uses professional-grade spatial analysis for sub-pixel collision detection.

## 📝 License

Educational Open Source. Feel free to use this to demonstrate Evolutionary Computation or Reinforcement Learning concepts.

---

**Build, Evolve, and Optimize! 🧬✨**
