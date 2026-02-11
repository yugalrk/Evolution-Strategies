import os
import json
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from es_pathfinder import solve_path

# --- Configuration ---
app = Flask(__name__, static_folder='.')
app.config['SECRET_KEY'] = 'es_secret_key_2024'

# Initialize SocketIO with CORS support and eventlet for production performance
# We use 'eventlet' to handle multiple concurrent real-time streams on Render
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')


# --- SocketIO Event Handler ---

@socketio.on('solve_path')
def handle_solve_path(json_data):
    """
    Receives map parameters from the client and starts the ES solver, 
    streaming updates back to the browser in real-time.
    """
    print(f"\n{'='*60}")
    print(f"🧬 Starting Evolution Strategy Pathfinding")
    print(f"{'='*60}")
    print(f"Source: {json_data['source']}")
    print(f"Destination: {json_data['destination']}")
    print(f"Obstacles: {len(json_data['obstacles'])} obstacle(s)")
    print(f"Waypoints: {json_data['num_waypoints']}")
    print(f"Generations: {json_data['generations']}")
    print(f"Initial Sigma: {json_data['initial_sigma']}")
    print(f"{'='*60}\n")
    
    # Call the ES solver function with socketio for real-time updates
    # Ensure es_pathfinder.py uses socketio.emit to send updates
    final_result = solve_path(json_data, socketio=socketio)
    
    # Send the final result back to the client
    emit('path_update', final_result)
    
    print(f"\n{'='*60}")
    print(f"✅ Evolution Complete!")
    print(f"Final Cost: {final_result['cost']:.4f}")
    print(f"Path Length: {final_result['length']:.4f}")
    print(f"Total Improvements: {final_result.get('improvements', 'N/A')}")
    print(f"{'='*60}\n")


# --- Flask Routing ---

@app.route('/')
def index():
    """Serves the main HTML client file."""
    return send_from_directory('.', 'index.html')


# --- Server Startup ---

if __name__ == '__main__':
    # 1. Grab the port from Render's environment, default to 5000 for local testing
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*70)
    print("🚀 EVOLUTION STRATEGY PATHFINDER SERVER")
    print(f"🌐 Running on: http://0.0.0.0:{port}")
    print("="*70)
    
    # 2. Host must be '0.0.0.0' so the external world can connect
    # 3. Debug must be False in production (Render)
    socketio.run(app, debug=False, host='0.0.0.0', port=port)