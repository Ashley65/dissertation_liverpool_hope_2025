import time
import random
import threading
import socket as sk
from AIModule import load_model
import torch


class AINetwork:
    def __init__(self, host, port, callback ):
        """
        Initialize the AI Network module.
        :param host: The address to bind the socket to.
        :param port: The port to bind the socket to.
        :param callback: Function to call when a message is received.
        """

        self.host = host
        self.port = port
        self.callback = callback
        self.running = True
        self.server_thread = threading.Thread(target=self.server_loop)

    def start(self):
        """
        Start the server thread.
        """
        self.server_thread.start()


    def stop(self):
        """
        Stop the server thread.
        """
        self.running = False
        self.server_thread.join()


    def server_loop(self):
        """
        Main server loop.
        """
        server_socket = sk.socket(sk.AF_INET, sk.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        print("Server listening on %s:%d" % (self.host, self.port))

        server_socket.settimeout(1.0)
        while self.running:
            try:
                client_socket, addr = server_socket.accept()
                data = client_socket.recv(1024).decode()
                if data:
                    # Invoke the callback function with the received message.
                    self.callback(data)
                client_socket.close()
            except sk.timeout:
                continue
        server_socket.close()


class AICaptain:
    def __init__(self, model_path=None, assessment_interval=10, network_port=9999):
        """
        Initialize the AI Captain.
        - assessment_interval: Time in seconds between periodic assessments (demo value).
        - network_port: Port number on which the network module listens.
        """
        self.assessment_interval = assessment_interval
        self.last_assessment = time.time()
        self.running = True

        # Load the AI model
        self.model = load_model(model_path)

        # Start networking module
        self.network = AINetwork("localhost", network_port, self.handle_network_message)
        self.network.start()


    def gather_data(self):
        """
        Simulate gathering sensor and telemetry data.
        In practice, replace this with actual sensor interfaces or simulation APIs.
        """
        return torch.tensor([
            random.uniform(0, 100),  # Fuel level
            random.uniform(0, 10),  # Gravity
            random.choice([0, 0, 0, 1])  # Anomaly (binary)
        ], dtype=torch.float32).unsqueeze(0)  # Convert to tensor for NN

    def strategic_planning(self, data):
        """
        Basic strategic planning that adjusts the mission plan based on sensor data.
        Replace with advanced logic as needed.
        """
        with torch.no_grad():
            prediction = self.model(data)  # Run inference
            action_index = torch.argmax(prediction).item()  # Get best action

        actions = [
            "Maintain current trajectory.",
            "Divert to nearest refueling station.",
            "Investigate unexpected anomaly."
        ]
        selected_action = actions[action_index]

        print(f"[AI Captain] New Strategy: {selected_action}")
        return selected_action

    def periodic_assessment(self):
        """Perform a periodic reassessment of the strategy."""
        print("\n[AI Captain] Performing periodic reassessment...")
        data = self.gather_data()
        self.strategic_planning(data)
        self.last_assessment = time.time()

    def event_driven_assessment(self, event_reason):
        """Trigger an event-driven reassessment based on a specific event."""
        print(f"\n[AI Captain] Event-driven reassessment triggered due to: {event_reason}")
        data = self.gather_data()
        self.strategic_planning(data)

    def handle_network_message(self, message):
        """
        Handle incoming network messages.
        This callback function checks the content of the message and triggers events accordingly.
        """
        print("\n[AI Captain] Received network message:", message)
        if "anomaly" in message.lower():
            self.event_driven_assessment("network event: anomaly detected")
        elif "update" in message.lower():
            print("[AI Captain] Processing network update message.")
        else:
            print("[AI Captain] Unknown network message received.")

    def run(self):
        """Main loop for the AI Captain that checks for periodic reassessment."""
        try:
            while self.running:
                current_time = time.time()
                if current_time - self.last_assessment >= self.assessment_interval:
                    self.periodic_assessment()
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop the AI Captain and the network module."""
        self.running = False
        self.network.stop()
        print("\n[AI Captain] Shutting down.")

# A simple client function to simulate sending a network message to the AI Captain.
def send_network_message(message, host="localhost", port=9999):
    try:
        client_socket = sk.socket(sk.AF_INET, sk.SOCK_STREAM)
        client_socket.connect((host, port))
        client_socket.sendall(message.encode())
        client_socket.close()
    except Exception as e:
        print(f"Failed to send network message: {e}")






if __name__ == "__main__":
    # Start the AI Captain with a demo assessment interval and network port.
    captain = AICaptain(assessment_interval=10, network_port=9999)
    captain_thread = threading.Thread(target=captain.run)
    captain_thread.start()

    # For demonstration, send a network message after 5 seconds.
    time.sleep(5)
    send_network_message("Event: anomaly detected!")
    # Let the system run for a short while to observe periodic and network-triggered assessments.
    time.sleep(15)
    captain.stop()
    captain_thread.join()
