import zmq
import json
import time
import random


class NetworkCommunication:
    """
    This class is responsible for handling network communication between the AI Captain and other systems
    """

    def __init__(self, server_address="tcp://localhost:5555"):
        """
        Initialise the NetworkCommunication class.
        :param server_address: The address to connect to the server.
        """
        self.server_address = server_address
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.server_address)

    def send_status(self, status):
        try:
            message = json.dumps({"type": "status_update", "data": status})
            self.socket.send_string(message)

            # Wait for acknowledgement from the server
            response = self.socket.recv_string()
            print(f"[AI] Status Acknowledged: {response}")
        except Exception as e:
            print(f"[AI] Error sending status: {e}")

    def receive_command(self):
        """Request and receive mission commands from the simulation."""
        try:
            self.socket.send_string(json.dumps({"type": "request_command"}))
            response = self.socket.recv_string()

            command_data = json.loads(response)
            print(f"[AI] Received Command: {command_data}")

            return command_data.get("command", None)  # Extract command

        except Exception as e:
            print(f"[AI] Error receiving command: {e}")
            return None


if __name__ == "__main__":
    ai_communication = NetworkCommunication()

    for _ in range(5):  # Simulating multiple interactions
        # Simulate AI sending status
        status = {
            "fuel_level": round(random.uniform(0.1, 1.0), 2),
            "velocity": [round(random.uniform(-1, 1), 2) for _ in range(3)],
            "position": [round(random.uniform(-100, 100), 2) for _ in range(3)],
            "anomaly_detected": random.choice([True, False])
        }
        ai_communication.send_status(status)

        # Receive and handle command
        command = ai_communication.receive_command()
        if command:
            print(f"[AI] Executing command: {command}")

        time.sleep(1)  # Simulate time