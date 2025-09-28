import zmq
import json
import random
import time
import logging
from datetime import datetime


class MissionControlServer:
    """
    Simulates the spacecraft firmware and handles communication with the AI Captain.
    """

    def __init__(self, server_address="tcp://*:5555"):
        # ZeroMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(server_address)
        self.server_address = server_address

        # Track spacecraft state
        self.spacecraft_status = {
            "fuel_level": 1.0,
            "position": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "anomaly_detected": False,
            "mission_phase": "TRANSIT"
        }

        # Setup logging
        logging.basicConfig(
            filename='mission_control.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('MissionControl')

        # Command queues and available commands remain the same
        self.available_commands = [
            'maintain_course',
            'adjust_trajectory',
            'increase_velocity',
            'decrease_velocity',
            'investigate_anomaly',
            'refuel',
            'emergency_protocol',
            'return_to_base'
        ]

        # Celestial targets for mission generation
        self.celestial_targets = [
            "Mars", "Jupiter", "Europa", "Titan", "Enceladus",
            "Asteroid Belt", "Kuiper Belt", "Oort Cloud"
        ]

        # Mission templates for complex missions
        self.mission_templates = [
            "Proceed to asteroid {target} to collect samples and return to the mother ship.",
            "Navigate to {target} and conduct a survey of the region. Avoid radiation belts.",
            "Execute emergency protocol and return to base immediately.",
            "Establish stable orbit around {target} and prepare for landing sequence.",
            "Conduct sampling mission at {target} while maintaining communication with Earth."
        ]

        self.logger.info(f"Mission Control Server started on {server_address}")

    def handle_message(self):
        """Process incoming messages from the AI Captain using the new standardized format."""
        try:
            message = self.socket.recv_string()
            self.logger.debug(f"Received: {message}")

            data = json.loads(message)

            # Create a base response with timestamp
            response = {
                "timestamp": datetime.now().isoformat()
            }

            # Handle different message types
            if data.get("type") == "ping":
                # Handle ping requests for synchronization check
                response["status"] = "ok"
                self.logger.debug("Ping request received and acknowledged")

            elif data.get("type") == "get_telemetry":
                # Generate and send telemetry data
                telemetry = self.generate_telemetry()
                response["status"] = "ok"
                response["data"] = telemetry
                self.logger.debug("Telemetry request received and fulfilled")

            elif data.get("type") == "status_update":
                # Process status update from AI Captain
                status_data = data.get("data", {})

                # Log the status update
                self.logger.info(f"Status update received: Phase={status_data.get('mission_phase', 'UNKNOWN')}, " +
                                 f"Step={status_data.get('mission_step', 0)}")

                # Store or process the status data as needed
                # You might want to save this for monitoring or analysis

                response["status"] = "ok"
                response["message"] = "Status update received"

            elif data.get("type") == "command":
                # Process command from AI Captain
                command = data.get("command")
                params = data.get("parameters", {})

                # Execute the command in the simulation
                success = self.execute_command(command, params)

                if success:
                    response["status"] = "ok"
                    self.logger.info(f"Command executed: {command} with params {params}")
                else:
                    response["status"] = "error"
                    response["message"] = f"Failed to execute command: {command}"
                    self.logger.warning(f"Command execution failed: {command}")

            else:
                # Unknown message type
                response["status"] = "error"
                response["message"] = "Unknown message type"
                self.logger.warning(f"Unknown message type received: {data.get('type')}")

            # Send the response
            self.socket.send_string(json.dumps(response))

        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            try:
                error_response = {
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                self.socket.send_string(json.dumps(error_response))
            except:
                pass

    def generate_telemetry(self):
        """Generate realistic telemetry data for the AI Captain."""
        # Add some randomness to make it more realistic
        fuel_decrease = random.uniform(0.001, 0.003)
        self.spacecraft_status["fuel_level"] = max(0.0, self.spacecraft_status["fuel_level"] - fuel_decrease)

        # Update position based on velocity
        for i in range(3):
            self.spacecraft_status["position"][i] += self.spacecraft_status["velocity"][i] * 0.1

        # Random chance of anomaly
        if random.random() < 0.05:  # 5% chance per telemetry request
            self.spacecraft_status["anomaly_detected"] = True
            self.spacecraft_status["anomaly_score"] = random.uniform(0.5, 0.9)
            self.spacecraft_status["anomaly_position"] = [
                self.spacecraft_status["position"][0] + random.uniform(-1, 1),
                self.spacecraft_status["position"][1] + random.uniform(-1, 1),
                self.spacecraft_status["position"][2] + random.uniform(-1, 1)
            ]
        else:
            self.spacecraft_status["anomaly_detected"] = False
            self.spacecraft_status["anomaly_score"] = 0.0

        # Add a target position for navigation
        self.spacecraft_status["target_position"] = [
            random.uniform(10, 20),
            random.uniform(10, 20),
            random.uniform(10, 20)
        ]

        # Return a copy of the current status
        return self.spacecraft_status.copy()

    def execute_command(self, command, parameters):
        """Execute a command in the simulation with given parameters."""
        try:
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Track success status and reason
            success = False
            reason = ""

            # Command attempt tracking
            command_counts = getattr(self, 'command_counts', {})
            self.command_counts = command_counts
            attempt_count = command_counts.get(command, 0) + 1
            command_counts[command] = attempt_count

            # Command implementation with command-specific logic
            if command == "maintain_course":
                # No change to velocity
                success = True

            elif command == "adjust_trajectory":
                # Modify velocity based on parameters
                if "target_position" in parameters:
                    target = parameters["target_position"]
                    # Simple vector towards target
                    for i in range(3):
                        self.spacecraft_status["velocity"][i] = (target[i] - self.spacecraft_status["position"][
                            i]) * 0.01
                    success = True

            elif command == "increase_velocity":
                # Increase speed by 10%
                for i in range(3):
                    self.spacecraft_status["velocity"][i] *= 1.1
                # Extra fuel consumption
                self.spacecraft_status["fuel_level"] -= 0.01
                success = True

            elif command == "decrease_velocity":
                # Decrease speed by 10%
                for i in range(3):
                    self.spacecraft_status["velocity"][i] *= 0.9
                success = True

            elif command == "investigate_anomaly":
                # Check if there's actually an anomaly to investigate
                if not self.spacecraft_status["anomaly_detected"] and "position" not in parameters:
                    success = False
                    reason = "No significant anomaly detected"
                else:
                    # Move towards anomaly position if one exists
                    if self.spacecraft_status["anomaly_detected"] and "position" in parameters:
                        anomaly_pos = parameters["position"]
                        # Adjust velocity towards anomaly
                        for i in range(3):
                            self.spacecraft_status["velocity"][i] = (anomaly_pos[i] -
                                                                     self.spacecraft_status["position"][i]) * 0.02
                    # Clear the anomaly (it was investigated)
                    self.spacecraft_status["anomaly_detected"] = False
                    success = True

            elif command == "refuel":
                # Refuel the spacecraft
                amount = 0.3  # Default amount
                if parameters.get("amount") == "max":
                    amount = 1.0 - self.spacecraft_status["fuel_level"]
                self.spacecraft_status["fuel_level"] = min(1.0, self.spacecraft_status["fuel_level"] + amount)
                success = True

            elif command == "emergency_protocol":
                # Check if there's an actual emergency condition
                if not self.spacecraft_status["anomaly_detected"] and self.spacecraft_status["anomaly_score"] < 0.6:
                    success = False
                    reason = "No emergency conditions detected"
                else:
                    # Stop all movement and enter emergency mode
                    for i in range(3):
                        self.spacecraft_status["velocity"][i] = 0.0
                    self.spacecraft_status["mission_phase"] = "EMERGENCY"
                    success = True

            elif command == "return_to_base":
                # Set velocity towards origin (0,0,0)
                magnitude = 0.0
                for i in range(3):
                    self.spacecraft_status["velocity"][i] = -self.spacecraft_status["position"][i] * 0.01
                    magnitude += self.spacecraft_status["velocity"][i] ** 2

                # Normalize for consistent speed
                magnitude = magnitude ** 0.5
                if magnitude > 0:
                    for i in range(3):
                        self.spacecraft_status["velocity"][i] /= magnitude
                        self.spacecraft_status["velocity"][i] *= 0.5  # Set speed to 0.5

                success = True

            else:
                self.logger.warning(f"Unknown command: {command}")
                reason = "Unknown command"
                success = False

            # Log the outcome in the desired format
            phase = self.spacecraft_status.get("mission_phase", "TRANSIT")
            fuel = self.spacecraft_status["fuel_level"]
            anomaly = self.spacecraft_status["anomaly_detected"]

            if success:
                log_status = "SUCCESS"
                log_message = f"{timestamp} - {command} - {log_status} - Phase: {phase}, Fuel: {fuel:.2f}, Sensors: fuel_level={fuel}, anomaly_detector={anomaly}"
            else:
                log_status = "FAILURE"
                command_with_attempt = f"{command} (attempt {attempt_count})"
                log_message = f"{timestamp} - {command_with_attempt} - {log_status} - Reason: {reason}, Phase: {phase}, Fuel: {fuel:.2f}, Sensors: fuel_level={fuel}, anomaly_detector={anomaly}"

            # Log to both logger and a separate outcomes log file
            self.logger.info(log_message)
            with open("outcomes.log", "a") as outcomes_file:
                outcomes_file.write(log_message + "\n")

            return success

        except Exception as e:
            self.logger.error(f"Error executing command {command}: {e}")
            return False

    def run_forever(self):
        """Run the server indefinitely, handling messages as they arrive."""
        self.logger.info("Server now listening for messages...")
        while True:
            self.handle_message()


def main():
    """Initialize and run the mission control server."""
    server = MissionControlServer()
    server.run_forever()


if __name__ == "__main__":
    main()
