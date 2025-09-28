import zmq
import json
import time
import random
import numpy as np
from threading import Thread


class MissionControlServer:
    """
    Server to simulate mission control operations and communicate with the AI Captain.
    """

    def __init__(self, server_address="tcp://*:5555"):
        self.server_address = server_address
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self.server_address)

        # Track spacecraft state
        self.spacecraft_status = {
            "fuel_level": 1.0,
            "position": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "anomaly_detected": False
        }

        # Command queue for scheduled commands
        self.command_queue = []
        self.command_counter = 0

        # Available commands matching AI Captain's action space
        self.available_commands = [
            'maintain_course',
            'adjust_trajectory',
            'increase_velocity',
            'decrease_velocity',
            'investigate_anomaly',
            'refuel',
            'emergency_protocol'
        ]

    def handle_message(self):
        """Process incoming messages from the AI Captain."""
        try:
            message = self.socket.recv_string()
            print(f"[MC] Received: {message}")

            data = json.loads(message)

            if data.get("type") == "status_update":
                # Update our record of spacecraft status
                self.spacecraft_status = data.get("data")
                print(f"[MC] Spacecraft status updated: {self.spacecraft_status}")

                # Acknowledge the status update
                self.socket.send_string(json.dumps({"status": "acknowledged"}))

            elif data.get("type") == "request_command":
                # Send a command if one is queued, otherwise send empty response
                command = self.get_next_command()

                if command:
                    print(f"[MC] Sending command: {command}")
                    self.socket.send_string(json.dumps({"command": command}))
                else:
                    print("[MC] No command to send")
                    self.socket.send_string(json.dumps({}))
            else:
                print(f"[MC] Unknown message type: {data.get('type')}")
                self.socket.send_string(json.dumps({"status": "error", "message": "unknown message type"}))

        except Exception as e:
            print(f"[MC] Error processing message: {e}")
            try:
                self.socket.send_string(json.dumps({"status": "error", "message": str(e)}))
            except:
                pass

    def get_next_command(self):
        """Get the next command from the queue or decide whether to issue a new command."""
        # If we have queued commands, use those first
        if self.command_queue:
            return self.command_queue.pop(0)

        # Otherwise randomly decide whether to issue a command (30% probability)
        if random.random() < 0.3:
            # Select a command based on spacecraft status
            if self.spacecraft_status.get("fuel_level", 1.0) < 0.2:
                # Low fuel, prioritize refueling
                return "refuel"
            elif self.spacecraft_status.get("anomaly_detected", False):
                # Anomaly detected, investigate it
                return "investigate_anomaly"
            else:
                # Otherwise select a random command
                return random.choice(self.available_commands)

        # 70% of the time, send no command
        return None

    def queue_command(self, command):
        """Add a command to the queue to be sent later."""
        if command in self.available_commands:
            self.command_queue.append(command)
            print(f"[MC] Command queued: {command}")
            return True
        else:
            print(f"[MC] Invalid command: {command}")
            return False

    def run_forever(self):
        """Run the server indefinitely, handling messages as they arrive."""
        print(f"[MC] Mission Control Server started on {self.server_address}")
        while True:
            self.handle_message()


def main():
    """Initialize and run the mission control server."""
    server = MissionControlServer()

    # Start the server in a separate thread
    server_thread = Thread(target=server.run_forever)
    server_thread.daemon = True
    server_thread.start()

    # Interactive command prompt
    print("Mission Control Server Interactive Console")
    print("Commands: queue <command_name>, status, help, exit")

    while True:
        try:
            cmd = input("> ").strip()

            if cmd == "exit":
                print("Shutting down...")
                break
            elif cmd == "status":
                print(f"Spacecraft Status: {server.spacecraft_status}")
            elif cmd == "help":
                print("Available commands:")
                print("  queue <command_name> - Queue a command to send to the spacecraft")
                print("  status - Show current spacecraft status")
                print("  help - Show this help message")
                print("  exit - Exit the program")
                print("\nAvailable spacecraft commands:")
                for command in server.available_commands:
                    print(f"  {command}")
            elif cmd.startswith("queue "):
                command_name = cmd.split("queue ")[1].strip()
                if server.queue_command(command_name):
                    print(f"Command '{command_name}' queued successfully.")
                else:
                    print(f"Invalid command '{command_name}'.")
            else:
                print("Unknown command. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()