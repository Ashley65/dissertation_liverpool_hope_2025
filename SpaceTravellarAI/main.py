import json
import logging
import os
import random
import time
from collections import deque

import numpy as np
import tensorflow as tf

from AIOptimser import AIOptimiser
from AnomalyDetector import AnomalyDetector, EnvironmentalHazardDetector, EnsembleAnomalyDetector
from MissionPhaseAwareness import MissionPhaseAwareness
from TransformerModel import DynamicTransformerModel, MissionPhase
from astroDataManager import AstroDataManager
from networking import NetworkCommunication, CommandHandler
from optimiser import FuelOptimiser
from sensor import EnhancedSensorData

# Configure TensorFlow GPU settings
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

logging.basicConfig(filename='ai_captain.log', level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Using TensorFlow device: {tf.config.list_physical_devices('GPU')}")


def create_anomaly_detection_ensemble(input_dim=8):
    """
    Create an ensemble of different anomaly detectors.

    Args:
        input_dim: Dimension of input data

    Returns:
        Configured ensemble detector
    """
    # Create ensemble
    ensemble = EnsembleAnomalyDetector(voting_threshold=0.5)

    # Add standard anomaly detector
    standard_detector = AnomalyDetector(input_dim=input_dim)

    # Add environmental hazard detector
    hazard_detector = EnvironmentalHazardDetector(input_dim=input_dim)

    # Create a third detector with different architecture for diversity
    diverse_detector = AnomalyDetector(input_dim=input_dim, latent_dim=6, threshold_multiplier=2.8)

    # Add all detectors to ensemble
    ensemble.add_detector(standard_detector)
    ensemble.add_detector(hazard_detector)
    ensemble.add_detector(diverse_detector)

    return ensemble


def train_ensemble(ensemble, normal_data, hazard_data=None, mission_step=None):
    """
    Train all detectors in the ensemble.

    Args:
        ensemble: The ensemble detector
        normal_data: Normal training data
        hazard_data: Optional hazard training data
        mission_step: Current mission step

    Returns:
        Trained ensemble detector
    """
    # Train each detector in the ensemble
    for i, detector in enumerate(ensemble.detectors):
        if isinstance(detector, EnvironmentalHazardDetector) and hazard_data:
            detector.train_on_environment_data(normal_data, hazard_data, mission_step)
        else:
            detector.train(normal_data, mission_step=mission_step)

    return ensemble


def detect_with_ensemble(ensemble, data, mission_step=None, historical_data=None):
    """
    Detect anomalies using the ensemble with adaptive thresholds.

    Args:
        ensemble: Trained ensemble detector
        data: Data to check for anomalies
        mission_step: Current mission step
        historical_data: Recent data for adaptive thresholds

    Returns:
        (is_anomaly, confidence_score, detection_result_dict)
    """
    # Apply adaptive thresholds if historical data is provided
    if historical_data is not None and len(historical_data) > 100:
        for detector in ensemble.detectors:
            # Update detector's threshold based on recent data patterns
            adaptive_threshold = detector.calculate_adaptive_threshold(
                historical_data, window_size=100
            )
            # Store the updated threshold
            detector.anomaly_threshold = adaptive_threshold

    # Get ensemble decision
    is_anomaly, confidence = ensemble.detect_anomaly(data, mission_step)

    # Get individual detector results for detailed analysis
    detector_results = {}
    for i, detector in enumerate(ensemble.detectors):
        detector_name = f"detector_{i}"
        if isinstance(detector, EnvironmentalHazardDetector):
            detector_name = "hazard_detector"
            is_hazard, hazard_score, hazard_type = detector.detect_hazard(data, mission_step=mission_step)
            detector_results[detector_name] = {
                "is_anomaly": is_hazard,
                "score": hazard_score,
                "hazard_type": hazard_type
            }
        else:
            is_detector_anomaly, detector_score = detector.detect_anomaly(data, mission_step)
            detector_results[detector_name] = {
                "is_anomaly": is_detector_anomaly,
                "score": detector_score
            }

    return is_anomaly, confidence, detector_results



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
        self.mission_memory = MissionMemory(max_memory_size=20000)
        self.network_comm = NetworkCommunication()
        self.command_handler = CommandHandler()


        # Feature dimension based on sensor data
        sample_features = self.sensor_data.preprocess_for_model(self.sensor_data.get_readings())
        self.input_dim = len(sample_features)

        # Initialise the anomaly detector
        self.anomaly_detector = AnomalyDetector(self.input_dim)

        self.hazard_detector = EnvironmentalHazardDetector(self.input_dim)
        self.hazard_detector_trained = False
        # Create an ensemble of anomaly detectors
        self.anomaly_detector = create_anomaly_detection_ensemble(self.input_dim)


        # Initialise the Dynamic Transformer model
        self.model = DynamicTransformerModel(self.input_dim, self.num_actions, max_sequence_length=25)

        # RL parameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995


        self.astro_data = AstroDataManager(data_dir="./data")
        self.fuel_optimiser = FuelOptimiser(self.astro_data, initial_fuel_level=1.0)

        self.ai_optimiser = AIOptimiser(self)

        # Initialise state history with max possible size
        self.state_history = deque(maxlen=self.model.max_sequence_length)

        # Initialise state history with zeros
        for _ in range(self.model.max_sequence_length):
            self.state_history.append(np.zeros(self.input_dim))

        self.mission_step = 0
        self.phase_awareness = MissionPhaseAwareness()
        self.current_phase = "TRANSIT"  # Default phase

        # Track normal data for anomaly detection training
        self.normal_data_buffer = []
        self.anomaly_detector_trained = False

    def train_anomaly_detector(self, force=False):
        """Train the anomaly detector on collected normal data."""
        if (len(self.normal_data_buffer) >= 200 and not self.anomaly_detector_trained) or force:
            normal_data = np.array(self.normal_data_buffer)
            logging.info(f"Training anomaly detector ensemble on {len(normal_data)} samples")

            # Train the ensemble using the train_ensemble function
            train_ensemble(self.anomaly_detector, normal_data, mission_step=self.mission_step)

            # Set the trained flag
            self.anomaly_detector_trained = True

            # Save each model in the ensemble
            self.anomaly_detector.save_ensemble("anomaly_detector_ensemble")
            return True
        return False

    def update_anomaly_detector(self, recent_data, mission_phase):
        """Update anomaly detector based on recent normal operational data"""
        # Only update if we have a trained model to start with
        if not self.anomaly_detector_trained:
            return False

        # Filter out any known anomalous data points
        normal_samples = []
        for experience in recent_data:
            # Extract the state from the experience tuple (state, action, reward, next_state, done)
            data_point = experience[0]  # Get just the state

            # Use ensemble to detect anomalies
            is_anomaly, score, _ = detect_with_ensemble(self.anomaly_detector, np.array([data_point]))
            # Only use data with low anomaly scores as new normal examples
            if not is_anomaly and score < 0.3:  # Using a fixed threshold for simplicity
                normal_samples.append(data_point)

        # Only proceed if we have enough filtered normal samples
        if len(normal_samples) < 10:
            logging.debug(f"Not enough normal samples ({len(normal_samples)}) for ensemble update")
            return False

        # Create a numpy array from our samples
        normal_data = np.array(normal_samples)

        # Update each detector in the ensemble
        for detector in self.anomaly_detector.detectors:
            detector.transfer_learning_update(normal_data, epochs=10)

        logging.info(
            f"Updated anomaly detector ensemble with {len(normal_data)} new samples from {mission_phase} phase")
        return True

    def train_hazard_detector(self, force=False):
        """Train the environmental hazard detector on collected normal data."""
        if (len(self.normal_data_buffer) >= 200 and not self.hazard_detector_trained) or force:
            normal_data = np.array(self.normal_data_buffer)
            logging.info(f"Training hazard detector on {len(normal_data)} samples")

            # Find the hazard detector in the ensemble (if it exists)
            for detector in self.anomaly_detector.detectors:
                if isinstance(detector, EnvironmentalHazardDetector):
                    # Create sample hazard labels for different environmental aspects
                    hazard_labels = {
                        "radiation": normal_data[:50],
                        "temperature": normal_data[50:100],
                        "pressure": normal_data[100:150],
                        "toxicity": normal_data[150:200]
                    }

                    detector.train_on_environment_data(normal_data, hazard_labels, self.mission_step)
                    self.hazard_detector_trained = True
                    detector.save_model("hazard_detector_model")
                    return True

            # If no hazard detector found in the ensemble, train the standalone one
            if hasattr(self, 'hazard_detector'):
                # Create sample hazard labels
                hazard_labels = {
                    "radiation": normal_data[:50],
                    "temperature": normal_data[50:100],
                    "pressure": normal_data[100:150],
                    "toxicity": normal_data[150:200]
                }

                self.hazard_detector.train_on_environment_data(normal_data, hazard_labels, self.mission_step)
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
        for experience in recent_data:
            # Extract the state from the experience tuple (state, action, reward, next_state, done)
            data_point = experience[0]  # Get just the state

            # Check if it's a hazard
            is_hazard, score, _ = self.hazard_detector.detect_hazard(np.array([data_point]))

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
        Use the Transformer model for decision-making.
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

        # Adjust priorities based on the mission phase
        phase_name, phase_info = MissionPhase.determine_phase(readings, self.current_phase)

        # Get phase-specific policy recommendations
        policy_recommendations = self.phase_awareness.get_policy_recommendations()
        action_preferences = policy_recommendations.get("action_preferences", {})

        # First, check for critical commands
        next_command = self.command_handler.get_next_command(self.mission_step)
        if next_command:
            for action_id, action_name in self.actions.items():
                if action_name == next_command:
                    logging.info(f"Executing command from CommandHandler: {next_command}")
                    return action_id

        if hasattr(self, 'anomaly_detector_trained') and self.anomaly_detector_trained:
            is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(state)
            readings['anomaly_detector'] = is_anomaly
            readings['anomaly_score'] = anomaly_score

            # Define a default anomaly threshold directly instead of using getattr on the detector
            default_anomaly_threshold = 0.7
            anomaly_threshold = default_anomaly_threshold

            # Try to get the threshold from the detector if it exists
            if hasattr(self.anomaly_detector,
                       'anomaly_threshold') and self.anomaly_detector.anomaly_threshold is not None:
                anomaly_threshold = self.anomaly_detector.anomaly_threshold

            # Handle significant anomalies immediately
            if is_anomaly and anomaly_score > anomaly_threshold * 1.3:
                logging.warning(f"Critical anomaly detected: score={anomaly_score}")
                return self.action_ids["emergency_protocol"]

            # Handle moderate anomalies with investigation
            elif is_anomaly and anomaly_score > anomaly_threshold * 1.1:
                logging.info(f"Significant anomaly detected: score={anomaly_score}")
                return self.action_ids["investigate_anomaly"]

        # Check for hazards
        if hasattr(self, 'hazard_detector_trained') and self.hazard_detector_trained:
            is_hazard, hazard_score, hazard_type = self.hazard_detector.detect_hazard(state)

            # Get the hazard threshold with a safe default value
            hazard_threshold = getattr(self.hazard_detector, 'anomaly_threshold', 0.6)

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

        # Fall back to model with an epsilon-greedy approach
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
            logging.debug(f"Exploration: Chose random action {self.actions[action]}.")
        else:
            # Get proper sequence based on current sequence length
            state_sequence = np.array(list(self.state_history))
            q_values = self.model.predict(state_sequence)[0]

            # Ensure q_values has the correct length
            if len(q_values) != self.num_actions:
                logging.warning(f"Model output dimension mismatch: got {len(q_values)}, expected {self.num_actions}")
                q_values = q_values[:self.num_actions] if len(q_values) > self.num_actions else np.pad(
                    q_values, (0, self.num_actions - len(q_values)), 'constant')

            # Apply phase-specific action biasing
            if phase_name == "TRANSIT":
                # Bias toward efficient actions during transit
                transit_bias = np.zeros_like(q_values)
                transit_bias[self.action_ids["maintain_course"]] = 0.2
                q_values += transit_bias

            elif phase_name == "EXPLORATION":
                # Bias toward investigation during exploration
                explore_bias = np.zeros_like(q_values)
                explore_bias[self.action_ids["investigate_anomaly"]] = 0.3
                q_values += explore_bias

            # Apply action preferences from phase awareness
            for action_name, bias in action_preferences.items():
                if action_name in self.action_ids:
                    action_id = self.action_ids[action_name]
                    if action_id < len(q_values):
                        q_values[action_id] += bias

            action = np.argmax(q_values)

            # Ensure action is within valid range
            if action >= self.num_actions:
                logging.warning(f"Invalid action index {action}, constraining to valid range")
                action = action % self.num_actions

            logging.debug(f"Exploitation: Predicted Q-values {q_values}, chose action {self.actions[action]}.")

        # Add bounds checking for any emergency override actions too
        if self.current_phase == "EMERGENCY" and np.random.random() < 0.4:
            emergency_action = self.action_ids["emergency_protocol"]
            logging.info(f"EMERGENCY OVERRIDE: Switching action to {self.actions[emergency_action]}")
            action = emergency_action

            # Final safety check to ensure action is always valid
        if action >= self.num_actions or action < 0:
            logging.error(f"Action {action} out of bounds, defaulting to maintain_course")
            action = self.action_ids["maintain_course"]

        return action



    def train(self, batch_size=256):
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
        """
        Calculate a reward for the current state-action pair with advanced mission awareness.
        Considers mission phase, fuel efficiency, and time-based factors.
        """
        readings = self.sensor_data.current_readings
        reward = 0.1  # Base survival reward
        action_name = self.actions[action]
        # Get current thresholds from phase awareness system
        current_thresholds = self.phase_awareness.current_thresholds

        # Get phase-specific reward modifiers
        policy = self.phase_awareness.get_policy_recommendations()
        reward_modifiers = policy.get("reward_modifiers", {})

        # Rest of evaluate_action with modified thresholds...
        fuel_minimum = current_thresholds.get("fuel_critical", 0.15)
        anomaly_tolerance = current_thresholds.get("anomaly_detection", 0.7)

        # Apply reward modifiers based on phase policy
        if "fuel_efficiency" in reward_modifiers and action_name in ["maintain_course", "adjust_trajectory"]:
            reward *= reward_modifiers["fuel_efficiency"]

        if "speed" in reward_modifiers and action_name in ["increase_velocity", "decrease_velocity"]:
            reward *= reward_modifiers["speed"]

        if "safety" in reward_modifiers and action_name in ["emergency_protocol", "investigate_anomaly"]:
            reward *= reward_modifiers["safety"]

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

        # Phase-specific requirements
        fuel_minimum = phase_info["requirements"]["fuel_minimum"]
        anomaly_tolerance = phase_info["requirements"]["anomaly_tolerance"]

        # === FUEL MANAGEMENT WITH EFFICIENCY CONSIDERATIONS ===
        if fuel_level < fuel_minimum:
            if action_name == 'refuel':
                reward += 1.0 * (1.0 - (fuel_level / fuel_minimum))
            else:
                # Scaled penalty based on how far below minimum and criticality of phase
                urgency_factor = 1.5 if phase_name in ["EMERGENCY", "CRITICAL", "LAUNCH"] else 0.8
                reward -= urgency_factor * (1.0 - (fuel_level / fuel_minimum))
        elif action_name == 'refuel' and fuel_level > 0.8:
            # Higher penalty for unnecessary refuelling that wastes time
            reward -= 0.4

        # === FUEL EFFICIENCY REWARDS ===
        # Reward maintaining course when appropriate (most efficient)
        if action_name == 'maintain_course' and not is_anomaly and fuel_level > fuel_minimum:
            reward += 0.2

        # Penalise wasteful actions based on phase
        if action_name in ['increase_velocity', 'decrease_velocity'] and phase_name == "TRANSIT":
            # In transit, constant velocity is most efficient
            recent_velocity_changes = sum(1 for s, a, r, ns, d in
                                          list(self.mission_memory.memory)[-5:]
                                          if self.actions[a] in ['increase_velocity', 'decrease_velocity'])
            if recent_velocity_changes > 2:
                reward -= 0.2  # Penalty for too many velocity changes

        # === ANOMALY HANDLING WITH PHASE-SPECIFIC URGENCY ===
        if is_anomaly:
            # Scale response urgency based on phase
            phase_urgency = {
                "LAUNCH": 0.95,  # Critical during launch
                "CRITICAL": 0.9,
                "EMERGENCY": 0.7,  # Already in emergency, so slightly lower
                "TRANSIT": 0.6,
                "EXPLORATION": 0.4  # More acceptable during exploration
            }.get(phase_name, 0.6)

            if action_name == 'investigate_anomaly':
                reward += 0.8 + anomaly_score * 0.4 * phase_urgency
            elif action_name != 'emergency_protocol' and anomaly_score > anomaly_tolerance:
                # Stronger penalty for ignoring anomalies in critical phases
                reward -= (anomaly_score - anomaly_tolerance) * 1.5 * phase_urgency
        elif action_name == 'investigate_anomaly':
            reward -= 0.3  # Higher penalty for wasting time investigating when no anomaly

        # === ENVIRONMENTAL HAZARD HANDLING ===
        if is_hazard:
            if action_name == 'emergency_protocol' and hazard_score > 0.7:
                # Correctly responding to serious hazards
                reward += 1.0
            elif action_name == 'investigate_anomaly' and 0.3 < hazard_score <= 0.7:
                # Appropriate response to moderate hazards
                reward += 0.7
            elif action_name == 'adjust_trajectory' and hazard_type in ['radiation', 'toxicity']:
                # Good response to avoid these types of hazards
                reward += 0.5
            elif hazard_score > 0.7 and action_name not in ['emergency_protocol', 'investigate_anomaly']:
                # Penalize ignoring serious hazards
                reward -= 0.8


        # === TIME-BASED PENALTIES ===
        # Penalise actions that do not make progress based on mission phase expectations
        expected_progress = {
            "LAUNCH": 0.8,  # Expected to make rapid progress
            "TRANSIT": 0.5,  # Steady progress
            "EXPLORATION": 0.2,  # Slower progress expected
            "CRITICAL": 0.7,
            "EMERGENCY": 0.9  # Need to resolve quickly
        }.get(phase_name, 0.5)

        # Calculate actual progress based on recent position changes
        if len(self.mission_memory.memory) > 5:
            # Fixed: Access the position data in the preprocessed state arrays
            # Position data is at indices 4, 5, 6 in the preprocessed state
            recent_positions = [s[4:7] for s, _, _, _, _ in list(self.mission_memory.memory)[-5:]]

            if len(recent_positions) >= 2:  # Ensure we have at least 2 positions
                recent_travel = np.linalg.norm(recent_positions[-1] - recent_positions[0])

                # For high-urgency phases, penalise lack of progress
                if phase_name in ["LAUNCH", "CRITICAL", "EMERGENCY"]:
                    progress_score = min(1.0, recent_travel / expected_progress)
                    if progress_score < 0.7 and action_name not in ['emergency_protocol', 'refuel']:
                        reward -= 0.2 * (1 - progress_score)

        # === PHASE-SPECIFIC REWARDS WITH TIME CONSIDERATION ===
        if phase_name == "LAUNCH":
            if action_name == 'increase_velocity':
                time_factor = max(0, 1 - (mission_time / 2.0))  # Decays to zero after 2 hours
                reward += 0.5 * time_factor
            elif action_name == 'decrease_velocity':
                reward -= 0.5

        elif phase_name == "TRANSIT":
            if action_name == 'maintain_course':
                reward += 0.3
            elif action_name == 'adjust_trajectory':
                # Only reward trajectory adjustments if they are needed
                position_magnitude = np.linalg.norm(position)
                if position_magnitude > 1.0:  # If we are off course
                    reward += 0.2
                else:
                    reward -= 0.1  # Penalty for unnecessary adjustments

        elif phase_name == "EXPLORATION":
            if action_name == 'investigate_anomaly':
                reward += 0.4
            elif action_name == 'adjust_trajectory':
                reward += 0.2
            # Time efficiency: reward completing exploration in a reasonable time
            if mission_time > 20 and action_name not in ['refuel', 'emergency_protocol']:
                reward -= 0.05  # Small-time pressure to finish exploration

        elif phase_name == "CRITICAL":
            if action_name == 'emergency_protocol':
                reward += 0.8
            elif action_name in ['increase_velocity', 'decrease_velocity']:
                reward += 0.2
            # Critical time pressure
            reward -= 0.1  # Base time pressure for all actions in the critical phase

        elif phase_name == "EMERGENCY":
            if action_name == 'emergency_protocol':
                reward += 1.5
            elif action_name in ['investigate_anomaly', 'refuel']:
                reward += 0.6
            else:
                reward -= 0.4
            # Severe time pressure in emergency
            reward -= 0.2  # Stronger time pressure for all actions

        # === ENVIRONMENTAL FACTORS ===
        temp_deviation = abs(temperature - 22.0)
        if temp_deviation > 5.0:
            reward -= 0.1 * (temp_deviation - 5.0) ** 2 / 25.0

        # === SEQUENTIAL ACTION PATTERNS ===
        recent_actions = [self.actions.get(a, '') for s, a, r, ns, d in
                          list(self.mission_memory.memory)[-5:]]

        # Discourage action repetition except for 'maintain_course'
        if action_name != 'maintain_course' and recent_actions.count(action_name) >= 3:
            reward -= 0.3

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

    def check_astro_data_manager(self):
        """Diagnostic function to check AstroDataManager integration."""
        if not hasattr(self, 'astro_data'):
            logging.error("AstroDataManager not initialized")
            return False

        # Check if astro_data is initialized properly
        try:
            # Basic data retrieval test - use an existing method
            celestial_bodies = self.astro_data.celestial_objects
            if not celestial_bodies:
                logging.info("No celestial bodies loaded yet, but AstroDataManager is initialized")
            else:
                logging.info(f"AstroDataManager has {len(celestial_bodies)} celestial bodies loaded")

            # Check integration with phase awareness
            if not hasattr(self, 'phase_awareness'):
                logging.error("PhaseAwareness system not initialized")
                return False

            # Test the integration of astro data with phase awareness
            status_data = {
                "position": self.sensor_data.get_readings().get("position", np.zeros(3)),
                "mission_step": self.mission_step,
                "timestamp": time.time()
            }

            # Test if we can get some example body data
            if celestial_bodies:
                # Try to get the first celestial body as a test
                first_body_id = list(celestial_bodies.keys())[0]
                body_data = self.astro_data.get_body_data(first_body_id)
                if body_data:
                    logging.info(f"Successfully retrieved data for celestial body: {first_body_id}")

            # Try to compute some test waypoints
            if len(celestial_bodies) >= 2:
                body_ids = list(celestial_bodies.keys())
                test_waypoints = self.astro_data.compute_navigation_waypoints(
                    body_ids[0], body_ids[1], mission_phase=self.current_phase
                )
                if test_waypoints:
                    logging.info(f"Successfully computed navigation waypoints")

            integration_result = self.phase_awareness.integrate_astro_data(self.astro_data, status_data)
            if not integration_result:
                logging.warning("AstroDataManager integration with PhaseAwareness may have issues")

            # Print available phases
            phases = self.phase_awareness.get_available_phases()
            logging.info(f"Available phases in system: {phases}")
            logging.info(f"Current phase confidence map: {self.phase_awareness.phase_confidence}")

            return True
        except Exception as e:
            logging.error(f"Error checking AstroDataManager: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False

    def take_action(self):
        """
        Select and execute an action based on current state.

        Returns:
            tuple: (action_name, reward) - The name of the action taken and the reward received
        """
        # Get sensor readings and preprocess for the model
        readings = self.sensor_data.get_readings()
        current_state = self.sensor_data.preprocess_for_model(readings)

        # Select action (either from your get_action method or directly)
        action_id = self.get_action(current_state)
        action_name = self.actions[action_id]

        # Execute the action
        result = self.run_action(action_id)

        # Evaluate the action and get reward
        reward = self.evaluate_action(current_state, action_id)

        # Log the action
        logging.info(f"Action taken: {action_name}, reward: {reward}")

        return action_name, reward


    def run_mission_step(self):
        """Execute a single step of the mission."""
        self.mission_step += 1
        logging.info(f"Starting mission step {self.mission_step}")

        # Get sensor readings
        readings = self.sensor_data.get_readings()
        current_state = self.sensor_data.preprocess_for_model(readings)

        # Prepare status data for phase awareness system
        status_data = {
            "fuel_level": readings["fuel_level"],
            "position": readings["position"],
            "velocity": readings["velocity"],
            "mission_step": self.mission_step,
            "timestamp": time.time(),
            "target_position": readings.get("target_position", None),
            "current_command": self.command_handler.last_executed_command,
            "collision_warning": readings.get("collision_warning", False),
            "system_failures": readings.get("system_failures", [])
        }

        # Get optimization data if available
        optimization_data = None
        if hasattr(self, 'fuel_optimiser'):
            target_position = readings.get('target_position', status_data['position'])
            optimization_data = self.fuel_optimiser.optimise_trajectory(
                status_data['position'],
                target_position,
                status_data['fuel_level'],
                self.current_phase
            )

        # Update phase awareness and get current phase information
        phase_data = self.phase_awareness.update(status_data, optimization_data)
        self.current_phase = phase_data["current_phase"]

        # Collect normal operational data when in normal phases
        if self.current_phase == "TRANSIT" and readings.get('anomaly_score', 0) < 0.3:
            self.normal_data_buffer.append(current_state)

        # Try to train the detectors if we have enough data
        if len(self.normal_data_buffer) >= 200:
            if not self.anomaly_detector_trained:
                self.train_anomaly_detector()
            if not hasattr(self, 'hazard_detector_trained') or not self.hazard_detector_trained:
                self.train_hazard_detector()

        # Log phase information
        logging.info(
            f"Current mission phase: {self.current_phase} (confidence: {phase_data['phase_confidence'].get(self.current_phase, 0):.2f})")

        # Integrate with AstroDataManager if available
        if hasattr(self, 'astro_data'):
            self.phase_awareness.integrate_astro_data(self.astro_data, status_data)

        # Update fuel optimizer with phase-specific parameters if available
        if hasattr(self, 'fuel_optimiser'):
            optimizer_config = self.fuel_optimiser.get_config()
            adapted_config = self.phase_awareness.adapt_optimizer_params(optimizer_config)
            self.fuel_optimiser.update_config(adapted_config)

        # Choose the next action
        action = self.get_action(current_state)

        # Add phase information to action context
        action_name = self.actions[action]
        logging.info(f"Executing action: {action_name} in phase {self.current_phase}")

        # Execute the action and get feedback
        reward = self.evaluate_action(current_state, action)

        # Get the next state
        next_readings = self.sensor_data.get_readings()
        next_state = self.sensor_data.preprocess_for_model(next_readings)

        # Adjust reward based on phase-specific policy
        policy = self.phase_awareness.get_policy_recommendations()
        reward_modifiers = policy.get("reward_modifiers", {})

        # Apply phase-specific reward modifiers
        if action_name == "maintain_course" and "fuel_efficiency" in reward_modifiers:
            reward *= reward_modifiers["fuel_efficiency"]
        elif action_name in ["increase_velocity", "decrease_velocity"] and "speed" in reward_modifiers:
            reward *= reward_modifiers["speed"]
        elif action_name in ["emergency_protocol", "investigate_anomaly"] and "safety" in reward_modifiers:
            reward *= reward_modifiers["safety"]

        # Store the experience in mission memory
        done = False  # Set to True if mission is complete
        self.mission_memory.add_experience(current_state, action, reward, next_state, done)

        # Send status update to server
        status_with_phase = status_data.copy()
        status_with_phase["mission_phase"] = self.current_phase
        # Fix: Use .get() method with a default value to avoid KeyError
        status_with_phase["phase_confidence"] = phase_data["phase_confidence"].get(self.current_phase, 0)
        self.network_comm.send_status(status_with_phase)

        # Train the model
        if self.mission_step % 10 == 0:  # Train every 10 steps
            self.train()

            # Update anomaly detector with recent data
            recent_data = self.mission_memory.get_recent_history(n_steps=50)
            if recent_data:
                self.update_anomaly_detector(recent_data, self.current_phase)
                self.update_hazard_detector(recent_data, self.current_phase)

        # Decrease exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action_name, reward


# Run a simulation of the AI Captain
def run_simulation(steps=1000):
    """Run a simulation of the AI Captain for a specified number of steps."""
    captain = AICaptain()

    # Try to load the previous checkpoint if available
    captain.load_checkpoint()

    training_frequency = 5  # Train every 10 steps to batch training
    experiences = 0

    try:
        for step in range(steps):
            # Run the mission step
            action_name, reward = captain.run_mission_step()
            experiences += 1

            # Get additional information for detailed output
            readings = captain.sensor_data.current_readings
            phase_name = captain.current_phase
            epsilon = captain.epsilon

            # Check for anomalies and hazards
            anomaly_detected = readings.get('anomaly_detector', False)
            anomaly_score = readings.get('anomaly_score', 0)
            hazard_detected = readings.get('environmental_hazard', False)
            hazard_type = readings.get('hazard_type', 'None')
            fuel_level = readings.get('fuel_level', 1.0)

            # Display detailed action information in the command line
            print(f"Step {step + 1}: Action = {action_name}, Reward = {reward:.4f}, Phase = {phase_name}")
            print(f"  Parameters: Epsilon = {epsilon:.4f}, Fuel = {fuel_level:.2f}")

            if anomaly_detected:
                print(f"  ANOMALY DETECTED: Score = {anomaly_score:.4f}")
            if hazard_detected:
                print(f"  HAZARD DETECTED: Type = {hazard_type}, Score = {readings.get('hazard_score', 0):.4f}")

            # Empty line for better readability
            print()

            # Add a mission termination check if needed
            done = False  # Replace with actual termination check if needed

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
    captain = AICaptain()
    # Check if AstroDataManager is working properly
    if captain.check_astro_data_manager():
        print("AstroDataManager verified and working")
    else:
        print("Issues with AstroDataManager detected, check logs")

    run_simulation(400)  # Increased steps for better utilization