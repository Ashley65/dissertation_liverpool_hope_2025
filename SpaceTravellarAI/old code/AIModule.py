import torch
import torch.nn as nn
import torch.optim as optim

# === 1. MULTI-MODAL ENCODER (Embeddings for different inputs) === #
class AICaptainNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=16, output_size=3):
        super(AICaptainNN, self).__init__()
        super(AICaptainNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input → Hidden Layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden → Output Layer
        self.softmax = nn.Softmax(dim=1)  # Normalize outputs

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x



# The function is meant to load a trained model (or create a new one)
def load_model(model_path=None):
    model = AICaptainNN()
    if model_path:
        model.load_state_dict(torch.load(model_path))
        model.eval()
    return model


def train_model(model, training_data, labels, epochs=100, learning_rate=0.001, model_save_path=None):
    """
    Train the AI Captain neural network model.

    Args:
        model: The AICaptainNN model instance
        training_data: Tensor of input features [batch_size, input_size]
        labels: Tensor of target outputs [batch_size, output_size]
        epochs: Number of training iterations
        learning_rate: Learning rate for optimizer
        model_save_path: Path to save the trained model

    Returns:
        The trained model
    """
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(training_data)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Save the trained model if a path is provided
    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    return model


import zmq
import json
import time


import zmq
import json
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../server.log"),  # Save to file
        logging.StreamHandler()  # Print to console
    ]
)

class NetworkServer:
    """ZeroMQ server to handle AI communication."""

    def __init__(self, bind_address="tcp://*:5555"):
        """Initialize the ZeroMQ server."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)  # Reply socket
        self.socket.bind(bind_address)
        self.command_queue = ["adjust_course", "conserve_fuel", "scan_area", "return_to_base"]
        logging.info("[SERVER] Initialized and listening on %s", bind_address)

    def run(self):
        """Main server loop."""
        while True:
            try:
                message = self.socket.recv_string()
                logging.info("[SERVER] Received raw message: %s", message)

                # Parse JSON safely
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logging.error("[SERVER] Failed to decode JSON: %s", message)
                    self.socket.send_string(json.dumps({"error": "Invalid JSON"}))
                    continue

                if data.get("type") == "status_update":
                    logging.info("[SERVER] Status update received: %s", data['data'])
                    self.socket.send_string(json.dumps({"response": "Status received"}))

                elif data.get("type") == "request_command":
                    command = self.command_queue.pop(0) if self.command_queue else "standby"
                    response = json.dumps({"command": command})
                    logging.info("[SERVER] Sending command: %s", command)
                    self.socket.send_string(response)

                else:
                    logging.warning("[SERVER] Unknown message type: %s", data)
                    self.socket.send_string(json.dumps({"error": "Unknown message type"}))

            except Exception as e:
                logging.exception("[SERVER] Error: %s", str(e))

if __name__ == "__main__":
    server = NetworkServer()
    server.run()
