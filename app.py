import os
import json
from flask import Flask, send_from_directory, request # Added request
from flask_socketio import SocketIO, emit
from es_pathfinder import solve_path

# --- Configuration ---
app = Flask(__name__, static_folder='.')
app.config['SECRET_KEY'] = 'es_secret_key_2024'

# Initialize SocketIO with CORS support and eventlet for production performance
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Track active tasks per session
active_tasks = {}

# --- SocketIO Event Handler ---

@socketio.on('solve_path')
def handle_solve_path(json_data):
    """
    Receives map parameters from the client and starts the ES solver, 
    streaming updates back to the browser in real-time.
    """
    sid = request.sid
    active_tasks[sid] = True
    
    print(f"\n{'='*60}")
    print(f"🧬 Starting Evolution Strategy Pathfinding for SID: {sid}")
    print(f"{'='*60}")
    print(f"Source: {json_data['source']}")
    print(f"Destination: {json_data['destination']}")
    print(f"Obstacles: {len(json_data['obstacles'])} obstacle(s)")
    print(f"Waypoints: {json_data['num_waypoints']}")
    print(f"Generations: {json_data['generations']}")
    print(f"Initial Sigma: {json_data['initial_sigma']}")
    print(f"{'='*60}\n")
    
    # Call the ES solver function with sid and active_tasks for isolation
    final_result = solve_path(json_data, socketio=socketio, sid=sid, tasks_dict=active_tasks)
    
    if sid in active_tasks:
        del active_tasks[sid]
    
    # Send final result only to the requester
    emit('path_update', final_result, to=sid)
    
    print(f"\n{'='*60}")
    print(f"✅ Evolution Complete for {sid}!")
    print(f"Final Cost: {final_result['cost']:.4f}")
    print(f"Path Length: {final_result['length']:.4f}")
    print(f"Total Improvements: {final_result.get('improvements', 'N/A')}")
    print(f"{'='*60}\n")

@socketio.on('stop_evolution')
def handle_stop():
    sid = request.sid
    if sid in active_tasks:
        active_tasks[sid] = False
        print(f"🛑 User stopped evolution: {sid}")

# --- Flask Routing ---

@app.route('/')
def index():
    """Serves the main HTML client file."""
    return send_from_directory('.', 'index.html')


# --- Server Startup ---

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*70)
    print("🚀 EVOLUTION STRATEGY PATHFINDER SERVER")
    print(f"🌐 Running on: http://0.0.0.0:{port}")
    print("="*70)
    
    socketio.run(app, debug=False, host='0.0.0.0', port=port)