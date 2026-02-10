import json
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from es_pathfinder import solve_path

# --- Configuration ---
app = Flask(__name__, static_folder='.')
app.config['SECRET_KEY'] = 'es_secret_key_2024'
# Initialize SocketIO with CORS support
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


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
    final_result = solve_path(json_data, socketio=socketio)
    
    # Send the final result
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
    print("\n" + "="*70)
    print("🚀 EVOLUTION STRATEGY PATHFINDER SERVER")
    print("="*70)
    print("\n📋 PREREQUISITES:")
    print("   Make sure you have installed the required packages:")
    print("   pip install Flask Flask-SocketIO numpy shapely")
    print("\n🌐 SERVER STARTING:")
    print("   Open your browser and navigate to:")
    print("   ➡️  http://127.0.0.1:5000/")
    print("\n📖 HOW TO USE:")
    print("   1. Set Source point (green) - click after selecting 'Set Source'")
    print("   2. Set Destination point (blue) - click after selecting 'Set Destination'")
    print("   3. Draw Obstacles (red) - drag rectangles after selecting 'Draw Obstacles'")
    print("   4. Click 'Run Evolution' to watch AI find the optimal path!")
    print("\n💡 TIP:")
    print("   Watch the waypoints evolve in real-time as the algorithm optimizes!")
    print("="*70 + "\n")
    
    socketio.run(app, debug=False, host='127.0.0.1', port=5000, allow_unsafe_werkzeug=True)