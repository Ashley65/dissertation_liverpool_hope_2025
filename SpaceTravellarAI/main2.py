import os
import numpy as np
import tensorflow as tf
import random
import time
import logging
import json
from collections import deque

from AnomalyDetector import AnomalyDetector, EnvironmentalHazardDetector
from TransformerModel import DynamicTransformerModel, MissionPhase
from networking import NetworkCommunication, CommandHandler
from sensor import EnhancedSensorData
from AIOptimser import AIOptimiser
from astroDataManager import AstroDataManager
from optimiser import FuelOptimiser

# Configure TensorFlow GPU settings
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

logging.basicConfig(filename='ai_captain.log', level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Using TensorFlow device: {tf.config.list_physical_devices('GPU')}")



class MissionMemory:
    """Stores and retrieves historical mission data for contextual decision making."""

    def __init__(self, max_memory_size=10000):
        self.memory = deque(maxlen=max_memory_size)

    def add_experience(self, state, action, reward, next_state, done):
        """Store a new experience in the memory."""
        self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size=64):
        """Sample a random batch of experiences for training."""
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def get_recent_history(self, n_steps=10):
        """Get the most recent mission history."""
        return list(self.memory)[-n_steps:] if len(self.memory) >= n_steps else list(self.memory)





class AICaptain:
    """Main AI Captain class that orchestrates the entire autonomous decision system."""

    def __init__(self, num_actions=8, gamma=0.95):
        # Define action space
        # Map actions to their names for lookups
        self.actions = {
            0: "maintain_course",
            1: "adjust_trajectory",
            2: "increase_velocity",
            3: "decrease_velocity",
            4: "investigate_anomaly",
            5: "refuel",
            6: "emergency_protocol",
            7: "return_to_base"
            # Add any other actions
        }
        self.action_ids = {v: k for k, v in self.actions.items()}

        self.num_actions = len(self.actions)

        # Initialize components
        self.sensor_data = EnhancedSensorData(self)
        self.mission_memory = MissionMemory()
        self.network_comm = NetworkCommunication()
        self.command_handler = CommandHandler()

        # Feature dimension based on sensor data
        sample_features = self.sensor_data.preprocess_for_model(self.sensor_data.get_readings())
        self.input_dim = len(sample_features)

        # Initialise the anomaly detector
        self.anomaly_detector = AnomalyDetector(self.input_dim)

        self.hazard_detector = EnvironmentalHazardDetector(self.input_dim)

        # Initialise the Dynamic Transformer model
        self.model = DynamicTransformerModel(self.input_dim, self.num_actions, max_sequence_length=25)

        # RL parameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995


        self.astro_data = AstroDataManager(data_dir="./data")
        self.fuel_optimiser = FuelOptimiser(self.astro_data, initial_fuel_level=1.0)

        self.ai_optimiser = AIOptimiser(self)

        # Initialise state history with max possible size
        self.state_history = deque(maxlen=self.model.max_sequence_length)

        # Initialise state history with zeros
        for _ in range(self.model.max_sequence_length):
            self.state_history.append(np.zeros(self.input_dim))

        self.mission_step = 0
        self.current_phase = "TRANSIT"  # Default phase

        # Track normal data for anomaly detection training
        self.normal_data_buffer = []
        self.anomaly_detector_trained = False

    def train_anomaly_detector(self, force=False):
        """Train the anomaly detector on collected normal data."""
        if (len(self.normal_data_buffer) >= 200 and not self.anomaly_detector_trained) or force:
            normal_data = np.array(self.normal_data_buffer)
            logging.info(f"Training anomaly detector on {len(normal_data)} samples")
            self.anomaly_detector.train(normal_data, epochs=50)
            self.anomaly_detector_trained = True
            self.anomaly_detector.save_model()
            return True
        return False

    def update_anomaly_detector(self, recent_data, mission_phase):
        """Update anomaly detector based on recent normal operational data"""
        # Only update if we have a trained model to start with
        if not self.anomaly_detector_trained:
            return False

        # Filter out any known anomalous data points
        normal_samples = []
        for data_point in recent_data:
            is_anomaly, score = self.anomaly_detector.detect_anomaly(data_point)
            # Only use data with low anomaly scores as new normal examples
            if not is_anomaly and score < (self.anomaly_detector.anomaly_threshold * 0.7):
                normal_samples.append(data_point)

        # Only proceed if we have enough filtered normal samples
        if len(normal_samples) < 10:
            logging.debug(f"Not enough normal samples ({len(normal_samples)}) for transfer learning update")
            return False

        # Create a numpy array from our samples
        normal_data = np.array(normal_samples)

        # Log the update operation
        logging.info(f"Updating anomaly detector with {len(normal_data)} new normal samples from {mission_phase} phase")

        # Use transfer learning to adapt the model to new normal patterns
        success = self.anomaly_detector.transfer_learning_update(normal_data, epochs=10)

        return success




    def train_hazard_detector(self, force=False):
        """Train the environmental hazard detector on collected normal data."""
        if (len(self.normal_data_buffer) >= 200 and not hasattr(self, 'hazard_detector_trained')) or force:
            normal_data = np.array(self.normal_data_buffer)
            logging.info(f"Training hazard detector on {len(normal_data)} samples")

            # Create sample hazard labels for different environmental aspects
            hazard_labels = {
                "radiation": normal_data[:50],  # Use a subset of data for each hazard type
                "temperature": normal_data[50:100],
                "pressure": normal_data[100:150],
                "toxicity": normal_data[150:200]
            }

            self.hazard_detector.train_on_environment_data(normal_data, hazard_labels)
            self.hazard_detector_trained = True
            self.hazard_detector.save_model("hazard_detector_model")
            return True
        return False

    def update_hazard_detector(self, recent_data, mission_phase):
        """Update the hazard detector based on recent normal operational data"""
        # Only update if we have a trained model to start with
        if not hasattr(self, 'hazard_detector_trained') or not self.hazard_detector_trained:
            return False

        # Filter out any known hazardous data points
        normal_samples = []
        for data_point in recent_data:
            is_hazard, score, _ = self.hazard_detector.detect_hazard(data_point)
            # Only use data with low hazard scores as new normal examples
            if not is_hazard and score < (self.hazard_detector.anomaly_threshold * 0.7):
                normal_samples.append(data_point)

        # Only proceed if we have enough filtered normal samples
        if len(normal_samples) < 10:
            logging.debug(f"Not enough normal samples ({len(normal_samples)}) for hazard detector update")
            return False

        # Create a numpy array from our samples
        normal_data = np.array(normal_samples)

        # Log the update operation
        logging.info(f"Updating hazard detector with {len(normal_data)} new normal samples from {mission_phase} phase")

        # Use transfer learning to adapt the model to new normal patterns
        success = self.hazard_detector.transfer_learning_update(normal_data, epochs=10)

        return success

    def get_action(self, state):
        """
        Determine the next action using either exploration or exploitation.
        Use the Transformer model for decision-making and integrate fuel and AI optimisers.
        """
        # Update state history
        self.state_history.append(state)

        # Get current sensor readings for phase determination
        readings = self.sensor_data.current_readings

        # Update a priority system with context-aware weights
        decision_priority = {
            "command_handler": 10,  # Earth commands take the highest priority
            "emergency": 9,  # Emergency conditions
            "anomaly": 7,  # Anomaly detection
            "hazard": 8,  # Environmental hazards
            "model": 5,  # Neural model predictions
            "exploration": 3  # Random exploration
        }

        # Determine mission phase
        phase_name, phase_info = MissionPhase.determine_phase(readings, self.current_phase)

        # In critical phases, increase anomaly and hazard priorities
        if phase_name in ["EMERGENCY", "CRITICAL", "LAUNCH"]:
            decision_priority["anomaly"] += 1
            decision_priority["hazard"] += 1

        # With low fuel, increase command handler priority
        if readings['fuel_level'] < phase_info["requirements"]["fuel_minimum"]:
            decision_priority["command_handler"] += 2

        # Get proper sequence based on current sequence length
        state_sequence = np.array(list(self.state_history))

        # First, check for critical commands
        next_command = self.command_handler.get_next_command(self.mission_step)
        if next_command:
            for action_id, action_name in self.actions.items():
                if action_name == next_command:
                    logging.info(f"Received command from Earth: {next_command}")
                    return action_id

        # Then, check anomalies with improved thresholding
        if hasattr(self, 'anomaly_detector_trained') and self.anomaly_detector_trained:
            is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(state)
            readings['anomaly_detector'] = is_anomaly
            readings['anomaly_score'] = anomaly_score

            # Handle significant anomalies immediately
            if is_anomaly and anomaly_score > self.anomaly_detector.anomaly_threshold * 1.3:
                logging.warning(f"Critical anomaly detected: score={anomaly_score}")
                return self.action_ids["emergency_protocol"]

            # Handle moderate anomalies with investigation
            elif is_anomaly and anomaly_score > self.anomaly_detector.anomaly_threshold * 1.1:
                logging.info(f"Significant anomaly detected: score={anomaly_score}")
                return self.action_ids["investigate_anomaly"]

        # Check for hazards
        if hasattr(self, 'hazard_detector_trained') and self.hazard_detector_trained:
            is_hazard, hazard_score, hazard_type = self.hazard_detector.detect_hazard(state)

            # Immediate response to critical hazards
            if is_hazard and hazard_score > 0.8:
                logging.warning(f"Critical hazard detected: {hazard_type}, score={hazard_score}")
                return self.action_ids["emergency_protocol"]

            # Response to significant hazards
            elif is_hazard and hazard_score > 0.6:
                if hazard_type in ["radiation", "toxicity"]:
                    logging.info(f"Avoiding {hazard_type} hazard: score={hazard_score}")
                    return self.action_ids["adjust_trajectory"]
                else:
                    logging.info(f"Investigating {hazard_type} hazard: score={hazard_score}")
                    return self.action_ids["investigate_anomaly"]

        # Use fuel optimiser to get fuel-efficient trajectory recommendations
        if hasattr(self, 'fuel_optimiser'):
            current_position = readings.get('position', [0.0, 0.0, 0.0])
            target_position = readings.get('target_position', [0.0, 0.0, 0.0])
            fuel_level = readings.get('fuel_level', 0.5)

            # Get optimisation recommendations
            optimisation = self.fuel_optimiser.optimise_trajectory(
                current_position, target_position, fuel_level, phase_name
            )

            # In critical fuel situations, prioritize fuel optimiser recommendations
            if fuel_level < 0.2 and optimisation.get('recommended_action') in self.action_ids:
                recommended_action = self.action_ids[optimisation.get('recommended_action')]
                logging.info(f"Low fuel: Using fuel optimiser recommendation: {self.actions[recommended_action]}")
                return recommended_action

        # Apply AI optimiser's action biases
        action_biases = {}
        if hasattr(self, 'ai_optimiser'):
            action_biases = self.ai_optimiser.get_action_biases(phase_name)

        # Fall back to model with an epsilon-greedy approach
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
            logging.debug(f"Exploration: Chose random action {self.actions[action]}.")
        else:
            q_values = self.model.predict(state_sequence)[0]

            # Ensure q_values has the correct length
            if len(q_values) != self.num_actions:
                logging.warning(f"Model output dimension mismatch: got {len(q_values)}, expected {self.num_actions}")
                q_values = q_values[:self.num_actions] if len(q_values) > self.num_actions else np.pad(
                    q_values, (0, self.num_actions - len(q_values)), 'constant')

            # Apply phase-specific action biasing from AI optimiser
            for action_name, bias in action_biases.items():
                if action_name in self.action_ids:
                    action_id = self.action_ids[action_name]
                    if action_id < len(q_values):
                        q_values[action_id] += bias

            action = np.argmax(q_values)
            logging.debug(f"Exploitation: Predicted Q-values {q_values}, chose action {self.actions[action]}.")

        # Emergency override
        if phase_name == "EMERGENCY" and np.random.random() < 0.4:
            emergency_action = self.action_ids["emergency_protocol"]
            logging.info(f"EMERGENCY OVERRIDE: Switching action to {self.actions[emergency_action]}")
            return emergency_action

        return action


    def train(self, batch_size=128):
        """Train the model using experiences from mission memory."""
        if len(self.mission_memory.memory) < batch_size:
            return

        experiences = self.mission_memory.sample_batch(batch_size)

        # Extract components from experiences
        state_sequences = []
        targets = []

        for state, action, reward, next_state, done in experiences:
            # Use full state history but will be adjusted inside the model
            state_history = list(self.state_history)[-(self.model.max_sequence_length - 1):] + [state]
            state_sequence = np.array(state_history)
            next_state_history = list(self.state_history)[-(self.model.max_sequence_length - 1) + 1:] + [next_state]
            next_state_sequence = np.array(next_state_history)

            # Get current Q-values prediction
            target = self.model.predict(state_sequence)[0]

            # Ensure target array has the correct length
            if len(target) != self.num_actions:
                logging.warning(f"Target length mismatch: {len(target)} vs {self.num_actions}")
                # Pad or truncate target to match num_actions
                if len(target) < self.num_actions:
                    target = np.pad(target, (0, self.num_actions - len(target)), 'constant')
                else:
                    target = target[:self.num_actions]

            # Ensure action is within bounds
            if action >= self.num_actions:
                logging.error(f"Invalid action index {action}, max allowed is {self.num_actions - 1}")
                continue

            if done:
                target[action] = reward
            else:
                next_q_values = self.model.predict(next_state_sequence)[0]
                # Ensure next_q_values has correct dimensions too
                if len(next_q_values) != self.num_actions:
                    next_q_values = next_q_values[:self.num_actions] if len(
                        next_q_values) > self.num_actions else np.pad(next_q_values,
                                                                      (0, self.num_actions - len(next_q_values)),
                                                                      'constant')
                target[action] = reward + self.gamma * np.max(next_q_values)

            state_sequences.append(state_sequence)
            targets.append(target)

        # Only proceed if we have samples to train on
        if len(targets) > 0:
            # Convert lists to numpy arrays before training
            targets = np.array(targets)
            state_sequences = np.array(state_sequences)

            # Train in larger batches for GPU efficiency
            self.model.train(state_sequences, targets)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def evaluate_action(self, state, action):
        """Calculate reward with advanced mission awareness and optimiser inputs."""
        readings = self.sensor_data.current_readings
        reward = 0.1  # Base survival reward
        action_name = self.actions[action]

        # Get the current mission phase
        phase_name, phase_info = MissionPhase.determine_phase(readings, self.current_phase)

        # Extract key readings
        fuel_level = readings['fuel_level']
        is_anomaly = readings.get('anomaly_detector', False)
        anomaly_score = readings.get('anomaly_score', 0.5 if is_anomaly else 0.0)
        temperature = readings['temperature']
        velocity = np.linalg.norm(readings['velocity'])
        mission_time = readings.get('mission_time', 0)
        position = readings.get('position', np.array([0.0, 0.0, 0.0]))

        is_hazard = readings.get('environmental_hazard', False)
        hazard_score = readings.get('hazard_score', 0)
        hazard_type = readings.get('hazard_type', None)

        # Use the FuelOptimiser to evaluate fuel efficiency
        if 'position' in readings:
            # Calculate a simplified target position (in reality, would come from mission data)
            target_position = position + readings.get('velocity', np.zeros(3)) * 10

            # Get optimisation insights from the fuel optimiser
            optimisation_result = self.fuel_optimiser.optimise_trajectory(
                current_position=position,
                target_position=target_position,
                current_fuel=fuel_level,
                phase=phase_name
            )

            # Reward fuel-efficient actions
            if action_name == optimisation_result["recommended_action"]:
                reward += 0.2
                logging.debug(f"Fuel optimization bonus: following recommended action {action_name}")

            # Add efficiency factor to reward
            efficiency_multiplier = optimisation_result.get("efficiency_multiplier", 1.0)
            reward *= (1.0 + (efficiency_multiplier - 1.0) * 0.5)
            logging.debug(f"Applied efficiency multiplier: {efficiency_multiplier}, adjusted reward: {reward}")

        # Use AstroDataManager for hazard-aware decisions
        if 'position' in readings:
            hazards = []

            # Find the nearest celestial body as a reference point
            nearest_body = None
            nearest_distance = float('inf')

            for body_id, body_data in self.astro_data.celestial_objects.items():
                body_position = np.array(body_data.get("position", [0, 0, 0]))
                distance = np.linalg.norm(position - body_position)

                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_body = body_id

            # If we have a nearest body, check for hazards
            if nearest_body:
                # Simulate a destination for path hazard check
                hazards = self.astro_data.get_hazards_near_path(
                    nearest_body,
                    nearest_body,  # We will use the same body when no specific target
                    safety_distance=2e6  # 2 million km safety distance
                )

                # Reward avoiding hazards or investigating them appropriately
                if hazards and action_name == "adjust_trajectory":
                    reward += 0.15
                    logging.debug(f"Hazard avoidance bonus: adjusting trajectory near {len(hazards)} hazard(s)")
                elif hazards and action_name == "investigate_anomaly":
                    reward += 0.1
                    logging.debug(f"Hazard investigation bonus: investigating near {len(hazards)} hazard(s)")

        # Update the AIOptimiser with the action performance
        self.ai_optimiser.update_from_experience(reward, action_name, phase_name)

        # Phase-specific requirements
        fuel_minimum = phase_info["requirements"]["fuel_minimum"]
        anomaly_tolerance = phase_info["requirements"]["anomaly_tolerance"]

        # === PHASE-SPECIFIC REWARDS WITH TIME CONSIDERATION ===
        if phase_name == "LAUNCH":
            if action_name == "increase_velocity":
                reward += 0.3
            elif action_name == "adjust_trajectory":
                reward += 0.2
            elif action_name == "emergency_protocol" and not is_anomaly:
                # Penalize unnecessary emergency protocols during launch
                reward -= 0.5

        elif phase_name == "TRANSIT":
            if action_name == "maintain_course":
                reward += 0.15
            elif action_name == "adjust_trajectory":
                # Consider distance to adjust optimal speed
                distance_factor = min(1.0, nearest_distance / 5e6) if 'nearest_distance' in locals() else 0.5
                reward += 0.2 * distance_factor

        elif phase_name == "EXPLORATION":
            if action_name == "investigate_anomaly" and is_anomaly:
                reward += 0.4
            elif action_name == "adjust_trajectory":
                # More rewards for adjustments when near celestial bodies
                if nearest_distance and nearest_distance < 3e6:
                    reward += 0.3
            # Time efficiency: reward completing exploration in a reasonable time
            if mission_time > 20 and action_name not in ['refuel', 'emergency_protocol']:
                reward -= 0.05 * (mission_time / 100)  # Graduated time pressure

        elif phase_name == "CRITICAL":
            if action_name == 'emergency_protocol':
                reward += 0.4 if is_anomaly or is_hazard else -0.3
            elif action_name in ['increase_velocity', 'decrease_velocity']:
                reward += 0.2 if is_hazard else 0.1
            # Critical time pressure
            reward -= 0.1  # Base time pressure for all actions in the critical phase

        elif phase_name == "EMERGENCY":
            if action_name == 'emergency_protocol':
                reward += 0.5
            elif action_name in ['investigate_anomaly', 'refuel']:
                reward += 0.3 if fuel_level < 0.3 else 0.2
            else:
                reward -= 0.1  # Penalize non-emergency actions
            # Severe time pressure in emergency
            reward -= 0.2  # Stronger time pressure for all actions

        # Global penalties for critical conditions
        if fuel_level < fuel_minimum:
            if action_name != "refuel":
                penalty = -0.3 * (fuel_minimum - fuel_level) / fuel_minimum
                reward += penalty
                logging.debug(f"Applied fuel shortage penalty: {penalty}")

        if is_anomaly and anomaly_score > anomaly_tolerance:
            if action_name not in ["investigate_anomaly", "emergency_protocol"]:
                penalty = -0.2 * (anomaly_score - anomaly_tolerance) / (1.0 - anomaly_tolerance)
                reward += penalty
                logging.debug(f"Applied anomaly ignorance penalty: {penalty}")

        if is_hazard and hazard_score > 0.5:
            if action_name not in ["adjust_trajectory", "emergency_protocol"]:
                penalty = -0.2 * hazard_score
                reward += penalty
                logging.debug(f"Applied hazard ignorance penalty: {penalty}")

        self.current_phase = phase_name
        return reward

    def save_checkpoint(self, filepath="ai_captain_checkpoint"):
        """Save the model and training state."""
        # Save the model
        self.model.save_model(filepath)

        # Save training state
        checkpoint = {
            'epsilon': self.epsilon,
            'mission_step': self.mission_step  # You'll need to add this counter to track progress
        }

        with open(f"{filepath}.json", "w") as f:
            json.dump(checkpoint, f)

        logging.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath="ai_captain_checkpoint"):
        """Load the model and training state."""
        # Load the model with proper extension handling
        if not self.model.load_model(filepath):
            logging.error(f"Failed to load model from {filepath}.keras or {filepath}.h5")
            return False

        # Load training state if it exists
        try:
            if os.path.exists(f"{filepath}.json"):
                with open(f"{filepath}.json", "r") as f:
                    checkpoint = json.load(f)

                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.mission_step = checkpoint.get('mission_step', 0)
                logging.info(f"Checkpoint loaded from {filepath}")
                return True
            else:
                logging.warning(f"No checkpoint state file found at {filepath}.json")
                return False
        except Exception as e:
            logging.error(f"Failed to load checkpoint state: {e}")
            return False

    def run_action(self, action):
        """
        Execute the action and determine its success based on current conditions.

        Args:
            action: The action ID to execute

        Returns:
            dict: Result information including success status and details
        """
        action_name = self.actions[action]
        readings = self.sensor_data.current_readings
        phase_name, phase_info = MissionPhase.determine_phase(readings, self.current_phase)

        # Default success - we'll modify this based on conditions
        success = True
        result_details = {}

        # Get environment conditions that might affect action success
        fuel_level = readings['fuel_level']
        distance_to_base = np.linalg.norm(readings.get('position', np.zeros(3)))
        is_anomaly = readings.get('anomaly_detector', False)
        anomaly_score = readings.get('anomaly_score', 0)
        is_hazard = readings.get('environmental_hazard', False)
        hazard_score = readings.get('hazard_score', 0)

        # Different actions have different success conditions
        if action_name == "refuel":
            # Refuel might fail if too far from base or in emergency conditions
            if distance_to_base > 10.0 and phase_name != "EMERGENCY":
                success = False
                result_details["reason"] = "Too far from refueling point"
            elif phase_name == "CRITICAL" and np.random.random() < 0.3:
                success = False
                result_details["reason"] = "Critical conditions prevented refueling"
            else:
                result_details["amount"] = min(0.5, 1.0 - fuel_level)

        elif action_name == "investigate_anomaly":
            # Investigation success depends on anomaly presence and conditions
            if not is_anomaly and anomaly_score < 0.3:
                success = False
                result_details["reason"] = "No significant anomaly detected"
            elif is_hazard and hazard_score > 0.7:
                success = False
                result_details["reason"] = "Hazard prevented safe investigation"

        elif action_name == "emergency_protocol":
            # Emergency protocols more likely to succeed in actual emergencies
            if not (is_anomaly and anomaly_score > 0.6) and not (is_hazard and hazard_score > 0.6):
                success = False
                result_details["reason"] = "No emergency conditions detected"

        elif action_name == "adjust_trajectory":
            # Trajectory adjustments might fail with low fuel
            if fuel_level < 0.15:
                success = fuel_level > 0.05 or np.random.random() < fuel_level * 2
                if not success:
                    result_details["reason"] = "Insufficient fuel for trajectory change"

        # Record environment state at time of action
        result_details.update({
            "phase": phase_name,
            "fuel_level": fuel_level,
            "timestamp": time.time()
        })

        if success:
            logging.info(f"Action '{action_name}' executed successfully")
        else:
            logging.warning(f"Action '{action_name}' failed: {result_details.get('reason', 'unknown reason')}")

        return {
            "success": success,
            "action": action_name,
            "details": result_details
        }

    def execute_command_with_tracking(self, action):
        """Execute a command and track its outcome."""
        action_name = self.actions[action]

        # Execute the command
        result = self.run_action(action)

        # Get the success value from the result
        success = result["success"]

        # Prepare details for outcome tracking
        details = {
            "mission_step": self.mission_step,
            "mission_phase": self.current_phase,
            "sensor_readings": {k: v for k, v in self.sensor_data.current_readings.items()
                                if k in ['anomaly_detector', 'fuel_level']}
        }

        # Add any additional details from the action result
        details.update(result.get("details", {}))

        # Record the outcome
        self.command_handler.record_outcome(action_name, success, details)

        # If a command failed and was critical, consider re-queuing it
        if not success and action_name in ["emergency_protocol", "refuel"]:
            self.command_handler.add_command(action_name,
                                             source="onboard_system",
                                             priority_override=9)  # High-priority retry

        return result

    def run_mission_step(self):
        """Execute a single step of the mission."""
        self.mission_step += 1
        logging.info(f"Starting mission step {self.mission_step}")

        # Get sensor readings and preprocess for the model
        readings = self.sensor_data.get_readings()
        current_state = self.sensor_data.preprocess_for_model(readings)

        # Create a status data package for network communication
        status_data = {
            "fuel_level": readings["fuel_level"],
            "position": readings["position"].tolist(),
            "velocity": readings["velocity"].tolist(),
            "mission_phase": self.current_phase,
            "mission_step": self.mission_step
        }

        # Add anomaly information if the detector is trained
        if hasattr(self, 'anomaly_detector_trained') and self.anomaly_detector_trained:
            is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(current_state)
            status_data["anomaly_detected"] = is_anomaly
            status_data["anomaly_score"] = float(anomaly_score)
        else:
            status_data["anomaly_detected"] = readings.get("anomaly_detector", False)

        # Add hazard information if detector is trained
        if hasattr(self, 'hazard_detector_trained') and self.hazard_detector_trained:
            is_hazard, hazard_score, hazard_type = self.hazard_detector.detect_hazard(current_state)
            status_data["hazard_detected"] = is_hazard
            status_data["hazard_score"] = float(hazard_score)
            status_data["hazard_type"] = hazard_type if hazard_type else "none"

        # Send status update to mission control
        self.network_comm.send_status(status_data)

        # Check for incoming commands from Earth
        command_data = self.network_comm.receive_command()
        if command_data:
            if command_data.get("type") == "complex_command":
                # Process complex mission command
                self.command_handler.add_mission(command_data["mission"], command_data["actions"])
                logging.info(
                    f"Received complex mission: {command_data['mission']['mission']} to {command_data['mission']['destination']}")
                logging.info(f"Action sequence: {command_data['actions']}")
            elif command_data.get("command"):
                # Process simple command
                logging.info(f"Received command from Earth: {command_data['command']}")


        action = None

        # Determine the next action (from Earth command or AI model)
        next_command = self.command_handler.get_next_command(self.mission_step)
        if next_command:
            # Convert command name to action ID
            if isinstance(next_command, dict) and "command" in next_command:
                next_command = next_command["command"]

            for action_id, action_name in self.actions.items():
                if action_name == next_command:
                    action = action_id
                    logging.info(f"Received command from Earth: {next_command}")
                    break
        else:
            # If no command from Earth, use an AI model to get the action
            action = self.get_action(current_state)

        # If no action was determined from commands, use AI model
        if action is None:
                # Use AI model to get the action
            action = self.get_action(current_state)


        # Execute the selected action
        logging.info(f"Executing action: {self.actions[action]}")
        action_result = self.execute_command_with_tracking(action)

        # Get the next state after action execution
        new_readings = self.sensor_data.get_readings()
        next_state = self.sensor_data.preprocess_for_model(new_readings)

        # Calculate reward for this action
        reward = self.evaluate_action(current_state, action)

        # Check for mission completion conditions
        done = False
        if readings['fuel_level'] <= 0.01:  # Critically low fuel
            logging.warning("MISSION CRITICAL: Out of fuel")
            done = True

        # Store experience for model training
        self.mission_memory.add_experience(current_state, action, reward, next_state, done)

        # Add current state to normal data buffer if no anomaly detected
        if not readings.get("anomaly_detector", False) and len(self.normal_data_buffer) < 500:
            self.normal_data_buffer.append(current_state)

        # Periodic training and optimisation
        if self.mission_step % 10 == 0:
            self.train()

            # Train anomaly detector if not already trained
            if not hasattr(self, 'anomaly_detector_trained') or not self.anomaly_detector_trained:
                self.train_anomaly_detector()

            # Train hazard detector if not already trained
            if not hasattr(self, 'hazard_detector_trained') or not self.hazard_detector_trained:
                self.train_hazard_detector()

        # Update detectors periodically with new normal patterns
        if self.mission_step % 50 == 0 and self.mission_step > 100:
            recent_states = [s for s, _, _, _, _ in list(self.mission_memory.memory)[-50:]]
            if recent_states:
                self.update_anomaly_detector(recent_states, self.current_phase)
                self.update_hazard_detector(recent_states, self.current_phase)

        # Save checkpoint periodically
        if self.mission_step % 100 == 0:
            self.save_checkpoint()

        return done


# Run a simulation of the AI Captain
def run_simulation(steps=1000):  # Increased steps for better utilisation
    """Run a simulation of the AI Captain for a specified number of steps."""
    captain = AICaptain()

    # Try to load the previous checkpoint if available
    captain.load_checkpoint()

    training_frequency = 10  # Train every 10 steps to batch training
    experiences = 0

    try:
        for step in range(steps):
            done = captain.run_mission_step()
            experiences += 1

            if step % training_frequency == 0 and experiences >= 64:
                print(f"\nStep {step + 1}/{steps} - Training on batch of experiences")
                captain.train(batch_size=64)
                experiences = 0

            if done:
                print("Mission complete or terminated!")
                break

            if step % 100 == 0:
                print(f"Step {step + 1}/{steps} completed. Epsilon: {captain.epsilon:.4f}")
                # Save checkpoint periodically
                captain.save_checkpoint()
    except Exception as e:
        logging.error(f"Error during simulation: {e}")
        # Save checkpoint on error to preserve training
        captain.save_checkpoint("ai_captain_emergency_checkpoint")
        raise

if __name__ == "__main__":
    run_simulation(200)  # Increased steps for better utilisation
