import json
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from es_pathfinder import solve_path

# --- Configuration ---
app = Flask(__name__, static_folder='.')
app.config['SECRET_KEY'] = 'es_secret_key'
# Initialize SocketIO, allowing connections from all origins
socketio = SocketIO(app, cors_allowed_origins="*")


# --- SocketIO Event Handler ---

@socketio.on('solve_path')
def handle_solve_path(json_data):
    """
    Receives map parameters from the client and starts the ES solver, 
    streaming updates back to the browser.
    """
    print(f"Received parameters for ES: {json_data}")
    
    # Call the Python solver function, passing the socketio object
    final_result = solve_path(json_data, socketio=socketio)
    
    # Send the final result path
    emit('path_update', final_result)
    print("Evolution finished and final path emitted.")


# --- Flask Routing ---

@app.route('/')
def index():
    """Serves the main HTML client file."""
    return send_from_directory('.', 'index.html')


# --- Server Run Command ---

if __name__ == '__main__':
    print("--- Starting Flask SocketIO Server ---")
    print("1. Ensure dependencies are installed: pip install Flask Flask-SocketIO numpy shapely")
    print("2. Open http://127.0.0.1:5000/ in your browser to interact.")
    socketio.run(app, debug=False, allow_unsafe_werkzeug=True)