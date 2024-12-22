from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# Initialize global model
global_model = torch.nn.Linear(28*28, 10)  # Example model
global_state = global_model.state_dict()

@app.route('/get_model', methods=['GET'])
def send_model():
    # Send the global model to clients
    return jsonify({k: v.tolist() for k, v in global_state.items()})

@app.route('/submit_update', methods=['POST'])
def receive_update():
    # Receive updates from clients
    global global_state
    client_update = request.get_json()
    client_update = {k: torch.tensor(v) for k, v in client_update.items()}
    
    # Aggregate updates (simple averaging for demonstration)
    for k in global_state:
        global_state[k] += client_update[k] / len(clients)  # Assume `clients` count known
    return "Update received", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
