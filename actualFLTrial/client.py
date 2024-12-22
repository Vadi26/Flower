import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Fetch global model from the server
def get_global_model(server_url):
    response = requests.get(f"{server_url}/get_model")
    model_state = response.json()
    return {k: torch.tensor(v) for k, v in model_state.items()}

# Send updated model to the server
def send_update(server_url, updated_state):
    payload = {k: v.tolist() for k, v in updated_state.items()}
    response = requests.post(f"{server_url}/submit_update", json=payload)
    return response.status_code

# Example training function
def train_local_model(model, train_loader, epochs, lr):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for _ in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

    return model.state_dict()

# Main client loop
if __name__ == "__main__":
    server_url = "http://192.168.1.11:5000"  # Replace <SERVER_IP> with server's IP
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    for round in range(5):  # Number of communication rounds
        # Step 1: Fetch global model
        global_model_state = get_global_model(server_url)
        model = torch.nn.Linear(28*28, 10)
        model.load_state_dict(global_model_state)

        # Step 2: Train locally
        local_state = train_local_model(model, train_loader, epochs=1, lr=0.1)

        # Step 3: Send updated model to the server
        status = send_update(server_url, local_state)
        print(f"Round {round + 1} completed, server response: {status}")
