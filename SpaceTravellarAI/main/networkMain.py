import logging
import numpy as np
import time
import os
import json
from datetime import datetime

# Import project components
from TransformerModel import DynamicTransformerModel, MissionMemorySystem
from networking import NetworkCommunication, CommandHandler, CommandParser
from sensor import EnhancedSensorData
from AIOptimser import AIOptimiser
from astroDataManager import AstroDataManager
from optimiser import FuelOptimiser
from MissionPhaseAwareness import MissionPhaseAwareness
from AnomalyDetector import AnomalyDetector, EnvironmentalHazardDetector, EnsembleAnomalyDetector

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NetworkedMission")
logger.setLevel(logging.INFO)


class NetworkedMissionControl:
    """
    Mission control class that receives environment data via network communication
    instead of defining the environment internally.
    """

    def __init__(self, server_address="tcp://localhost:5555"):
        # Initialize network communication
        self.network_comm = NetworkCommunication(server_address)

        # Initialize core components
        self.astro_data = AstroDataManager()
        self.command_handler = CommandHandler()
        self.phase_awareness = MissionPhaseAwareness()

        # Initialize sensor system with minimal config (will be updated from network)
        self.sensor_data = EnhancedSensorData(self._get_default_sensor_config())

        # Initialize memory system
        self.memory_system = MissionMemorySystem(max_episodes=200)

        # Initialize command parser
        self.command_parser = CommandParser()

        # Actions available to the AI
        self.actions = {
            0: "maintain_course",
            1: "adjust_trajectory",
            2: "increase_velocity",
            3: "decrease_velocity",
            4: "investigate_anomaly",
            5: "refuel",
            6: "emergency_protocol",
            7: "return_to_base"
        }
        self.action_to_id = {v: k for k, v in self.actions.items()}

        # Initialize fuel optimizer
        self.fuel_optimizer = FuelOptimiser(self.astro_data)

        # Initialize AI components
        self.init_ai_components()

        # Mission state
        self.mission_step = 0
        self.current_phase = "TRANSIT"
        self.anomaly_detected = False
        self.normal_data_buffer = []
        self.anomaly_detector_trained = False
        self.hazard_detector_trained = False

        # History tracking
        self.action_history = []
        self.reward_history = []
        self.state_history = []

        # Attempt to load previous models if available
        self.load_trained_components()

        logger.info("NetworkedMissionControl initialized and ready")

    def _get_default_sensor_config(self):
        """Provide a default sensor configuration."""
        return {
            "position_accuracy": 1000,  # meters
            "velocity_accuracy": 10,  # m/s
            "fuel_accuracy": 0.01,  # percentage points
            "sample_rate": 1,  # Hz
            "sensors": {
                "position": True,
                "velocity": True,
                "fuel": True,
                "temperature": True,
                "radiation": True,
                "magnetic": True
            }
        }

    def init_ai_components(self):
        """Initialize AI decision-making components."""
        # Input dimensions based on sensor data representation
        input_dim = 12  # Position (3), Velocity (3), Target (3), Fuel (1), Time (1), Phase (1)
        num_actions = len(self.actions)

        # Initialize transformer model for decision-making
        self.model = DynamicTransformerModel(input_dim=input_dim, num_actions=num_actions)

        # Initialize anomaly detection
        self.anomaly_detector = EnsembleAnomalyDetector()

        # Initialize AI optimizer
        self.ai_optimizer = AIOptimiser(self.model)

        # Exploration parameters
        self.epsilon = 0.3
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32

        # Experience buffer for training
        self.experience_buffer = []

    def load_trained_components(self):
        """Load previously trained models if available."""
        try:
            # Try to load transformer model
            if os.path.exists("models/mars_mission_network"):
                self.model.load_model("models/mars_mission_network")
                logger.info("Loaded pre-trained decision model")

            # Try to load anomaly detector
            if os.path.exists("models/anomaly_detector_network"):
                self.anomaly_detector.load_ensemble("models/anomaly_detector_network")
                self.anomaly_detector_trained = True
                logger.info("Loaded pre-trained anomaly detector")

            # Load mission metadata
            if os.path.exists("models/mission_metadata_network.json"):
                with open("models/mission_metadata_network.json", "r") as f:
                    metadata = json.load(f)
                    self.epsilon = metadata.get("epsilon", self.epsilon)
                    logger.info(f"Loaded mission metadata, epsilon set to {self.epsilon}")

            return True
        except Exception as e:
            logger.error(f"Error loading trained components: {str(e)}")
            return False

    def save_trained_components(self):
        """Save trained models to disk."""
        try:
            os.makedirs("models", exist_ok=True)

            # Save transformer model
            if hasattr(self.model, 'save_model'):
                self.model.save_model("models/mars_mission_network")

            # Save anomaly detector
            if hasattr(self.anomaly_detector, 'save_ensemble'):
                self.anomaly_detector.save_ensemble("models/anomaly_detector_network")

            # Save mission metadata
            metadata = {
                "mission_name": "network",
                "last_saved": datetime.now().isoformat(),
                "mission_step": self.mission_step,
                "epsilon": self.epsilon
            }

            with open("models/mission_metadata_network.json", "w") as f:
                json.dump(metadata, f)

            logger.info("Saved trained components successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving trained components: {str(e)}")
            return False

    def preprocess_sensor_data(self, sensor_readings):
        """Process sensor readings into model input features."""
        # Extract and normalize key features
        position = np.array(sensor_readings.get('position', [0, 0, 0])) / 1e11
        velocity = np.array(sensor_readings.get('velocity', [0, 0, 0])) / 1e5
        target_position = np.array(sensor_readings.get('target_position', [0, 0, 0])) / 1e11
        fuel_level = sensor_readings.get('fuel_level', 1.0)
        mission_time = sensor_readings.get('mission_time', 0) / 1000  # Normalize to 0-1

        # Create one-hot encoding for mission phase
        phase_encoding = 0.0  # Default encoding
        if 'mission_phase' in sensor_readings:
            phases = ["LAUNCH", "TRANSIT", "EXPLORATION", "CRITICAL", "EMERGENCY"]
            if sensor_readings['mission_phase'] in phases:
                phase_encoding = phases.index(sensor_readings['mission_phase']) / len(phases)

        # Combine features into single vector
        features = np.concatenate([
            position,  # 3 values
            velocity,  # 3 values
            target_position,  # 3 values
            [fuel_level],  # 1 value
            [mission_time],  # 1 value
            [phase_encoding]  # 1 value
        ])

        return features

    def detect_anomalies(self, sensor_data):
        """Use ML-based anomaly detection if trained, otherwise use network-provided values."""
        if not self.anomaly_detector_trained:
            # Use anomaly information from the network if available
            is_anomaly = sensor_data.get('anomaly_detected', False)
            confidence = sensor_data.get('anomaly_score', 0.0)
            return is_anomaly, confidence

        # Use trained detector for anomaly detection
        features = self.preprocess_sensor_data(sensor_data)
        is_anomaly, confidence = self.anomaly_detector.detect_anomaly(
            np.array([features]),
            self.mission_step
        )

        return is_anomaly, confidence

    def train_anomaly_detector(self):
        """Train anomaly detector on collected normal data."""
        if len(self.normal_data_buffer) < 200:
            logger.info(f"Not enough normal data ({len(self.normal_data_buffer)}/200) to train anomaly detector")
            return False

        normal_data = np.array(self.normal_data_buffer)
        logger.info(f"Training anomaly detector on {len(normal_data)} normal samples")

        # Train each detector in the ensemble
        for detector in self.anomaly_detector.detectors:
            if isinstance(detector, EnvironmentalHazardDetector):
                # Create sample hazard labels for environmental aspects
                hazard_labels = {
                    "radiation": normal_data[:50] if len(normal_data) >= 50 else normal_data,
                    "temperature": normal_data[50:100] if len(normal_data) >= 100 else normal_data[:50],
                    "pressure": normal_data[100:150] if len(normal_data) >= 150 else normal_data[:50],
                    "toxicity": normal_data[150:200] if len(normal_data) >= 200 else normal_data[:50]
                }
                detector.train_on_environment_data(normal_data, hazard_labels, self.mission_step)
            else:
                detector.train(normal_data, mission_step=self.mission_step)

        self.anomaly_detector_trained = True
        logger.info("Anomaly detector trained successfully")

        # Save detector thresholds
        self.save_trained_components()

        return True

    def select_action(self, state_features):
        """Select an action based on current state features."""
        # Exploration: random action with probability epsilon
        if np.random.random() < self.epsilon:
            action_id = np.random.choice(list(self.actions.keys()))
            action_name = self.actions[action_id]
            logger.info(f"Exploration: Selected random action: {action_name}")
            return action_name

        # Exploitation: use model if trained
        if hasattr(self.model, 'trained') and self.model.trained:
            # Add state to history for sequence input
            if len(self.state_history) < self.model.max_sequence_length:
                sequence = [state_features] * (
                            self.model.max_sequence_length - len(self.state_history)) + self.state_history
            else:
                sequence = self.state_history[-self.model.max_sequence_length:]
                if len(sequence) < self.model.max_sequence_length:
                    sequence = [state_features] * (self.model.max_sequence_length - len(sequence)) + sequence

            # Get Q-values from model
            q_values = self.model.predict(np.array([sequence]))[0]
            action_id = np.argmax(q_values)
            action_name = self.actions.get(action_id, "maintain_course")
            logger.info(f"Model prediction: {action_name} (Q-values: {q_values})")
            return action_name

        # Fallback heuristic policy if model not trained
        sensor_data = self.sensor_data.current_readings
        fuel_level = sensor_data.get('fuel_level', 1.0)

        if self.anomaly_detected:
            return "investigate_anomaly" if np.random.random() < 0.7 else "emergency_protocol"
        elif fuel_level < 0.2:
            return "refuel"
        else:
            # Basic rule-based selection
            if self.current_phase == "LAUNCH":
                return "increase_velocity"
            elif self.current_phase == "TRANSIT":
                return "maintain_course" if np.random.random() < 0.7 else "adjust_trajectory"
            elif self.current_phase == "EXPLORATION":
                exploration_actions = ["decrease_velocity", "adjust_trajectory", "investigate_anomaly"]
                return np.random.choice(exploration_actions)
            elif self.current_phase in ["CRITICAL", "EMERGENCY"]:
                emergency_actions = ["emergency_protocol", "decrease_velocity", "refuel"]
                return np.random.choice(emergency_actions)
            else:
                return "maintain_course"

    def train_model(self):
        """Train the decision model on collected experiences."""
        if len(self.experience_buffer) < self.batch_size:
            logger.info(f"Not enough experiences for training: {len(self.experience_buffer)}/{self.batch_size}")
            return False

        # Sample batch of experiences
        indices = np.random.choice(len(self.experience_buffer), self.batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]

        # Prepare training data
        state_sequences = []
        targets = []

        for state, action_id, reward, next_state, done in batch:
            # Create state sequence for input
            state_sequence = np.array(list(self.state_history)[-(self.model.max_sequence_length - 1):] + [state])

            # Get current Q-value predictions
            q_values = self.model.predict(state_sequence)[0]

            # Calculate target Q-values
            if done:
                q_values[action_id] = reward
            else:
                # Next state sequence
                next_state_sequence = np.array(
                    list(self.state_history)[-(self.model.max_sequence_length - 1) + 1:] + [next_state])
                next_q_values = self.model.predict(next_state_sequence)[0]
                q_values[action_id] = reward + 0.95 * np.max(next_q_values)  # gamma = 0.95

            state_sequences.append(state_sequence)
            targets.append(q_values)

        # Train model
        self.model.train(np.array(state_sequences), np.array(targets))
        logger.info(f"Trained model on batch of {len(batch)} experiences")

        # Save periodically
        if self.mission_step % 50 == 0:
            self.save_trained_components()

        return True

    def process_mission_step(self):
        """Process a single mission step using networked communication."""
        self.mission_step += 1
        logger.info(f"Starting mission step {self.mission_step}")

        # Request latest environment state from the network
        try:
            # Get sensor readings from network
            self.sensor_data.current_readings = self.network_comm.receive_state()

            if not self.sensor_data.current_readings:
                logger.error("Failed to receive sensor data from network")
                return False

            # Update mission phase
            self.current_phase = self.sensor_data.current_readings.get('mission_phase', 'TRANSIT')

            # Check for anomalies
            is_anomaly, confidence = self.detect_anomalies(self.sensor_data.current_readings)
            self.anomaly_detected = is_anomaly
            self.sensor_data.current_readings['anomaly_detector'] = is_anomaly
            self.sensor_data.current_readings['anomaly_score'] = confidence

            # Process sensor data for model input
            current_state = self.preprocess_sensor_data(self.sensor_data.current_readings)

            # Update state history
            self.state_history.append(current_state)
            if len(self.state_history) > 20:  # Keep last 20 states
                self.state_history.pop(0)

            # Collect normal operational data for anomaly detector training
            if not is_anomaly and self.current_phase == "TRANSIT" and confidence < 0.3:
                self.normal_data_buffer.append(current_state)
                if len(self.normal_data_buffer) > 500:
                    self.normal_data_buffer.pop(0)

            # Check if we should train the anomaly detector
            if len(self.normal_data_buffer) >= 200 and not self.anomaly_detector_trained:
                self.train_anomaly_detector()

            # Check for commands from mission control
            command_data = self.network_comm.receive_command()
            if command_data:
                if command_data["type"] == "complex_command":
                    # Process complex mission command
                    self.command_handler.add_mission(
                        command_data["mission"],
                        command_data["actions"]
                    )
                    logger.info(f"Received complex mission command: {command_data['mission']['mission']}")

                elif command_data["type"] == "simple_command":
                    # Process direct action command
                    self.command_handler.add_command(
                        command_data["command"],
                        source="mission_control"
                    )
                    logger.info(f"Received simple command: {command_data['command']}")

                elif command_data["type"] == "parameterized_command":
                    # Process parameterized command
                    cmd_data = command_data["command_data"]
                    self.command_handler.add_command(
                        cmd_data["command"],
                        source="mission_control",
                        parameters=cmd_data.get("parameters", {})
                    )
                    logger.info(f"Received parameterized command: {cmd_data['command']}")

            # Get next command from handler or select action with AI
            command = self.command_handler.get_next_command(self.mission_step)

            if not command:
                # No command from handler, use AI to select action
                action_name = self.select_action(current_state)
                logger.info(f"AI selected action: {action_name}")
            else:
                # Use command from handler
                action_name = command
                logger.info(f"Executing command from handler: {action_name}")

            # Execute the selected action by sending to simulation
            action_result = self.network_comm.send_action(action_name)

            if not action_result:
                logger.error(f"Failed to send action {action_name} to simulation")
                return False

            # Get action result and reward
            reward = action_result.get("reward", 0.0)
            success = action_result.get("success", False)

            # Record the outcome
            self.command_handler.record_outcome(
                action_name,
                success,
                {"reward": reward, "mission_phase": self.current_phase}
            )

            # Store experience for training
            action_id = self.action_to_id.get(action_name, 0)

            # Get next state after action
            next_sensor_data = self.network_comm.receive_state()
            if next_sensor_data:
                next_state = self.preprocess_sensor_data(next_sensor_data)
                done = next_sensor_data.get("mission_complete", False)

                # Add to experience buffer
                self.experience_buffer.append((current_state, action_id, reward, next_state, done))
                if len(self.experience_buffer) > 1000:  # Limit buffer size
                    self.experience_buffer.pop(0)

                # Update histories
                self.action_history.append(action_name)
                self.reward_history.append(reward)

                # Periodically train the model
                if len(self.experience_buffer) >= self.batch_size and self.mission_step % 5 == 0:
                    self.train_model()

                # Decrease exploration rate
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                return True
            else:
                logger.error("Failed to receive next state after action")
                return False

        except Exception as e:
            logger.error(f"Error in mission step: {str(e)}")
            return False

    def run(self, max_steps=1000):
        """Run the networked mission control for a specified number of steps."""
        steps_completed = 0
        try:
            for step in range(max_steps):
                success = self.process_mission_step()
                if not success:
                    logger.warning(f"Mission step {self.mission_step} failed, attempting to continue")
                    # Try to reconnect if needed
                    time.sleep(1)
                    continue

                steps_completed += 1

                # Check for mission completion
                if self.sensor_data.current_readings.get("mission_complete", False):
                    logger.info(f"Mission completed successfully after {steps_completed} steps!")
                    break

                # Save checkpoint periodically
                if step % 100 == 0:
                    self.save_trained_components()
                    logger.info(f"Checkpoint saved at step {steps_completed}")

                # Small delay to prevent overwhelming the network
                time.sleep(0.1)

            logger.info(f"Mission run completed with {steps_completed} steps")
            self.save_trained_components()
            return True

        except KeyboardInterrupt:
            logger.info("Mission interrupted by user")
            self.save_trained_components()
            return False
        except Exception as e:
            logger.error(f"Error during mission run: {str(e)}")
            self.save_trained_components()
            return False


if __name__ == "__main__":
    # Create and run the networked mission control
    mission = NetworkedMissionControl()
    mission.run(max_steps=1000)