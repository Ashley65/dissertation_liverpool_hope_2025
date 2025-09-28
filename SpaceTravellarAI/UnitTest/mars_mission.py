#!/usr/bin/env python3
# mars_mission.py

import logging
import numpy as np
import random
import time
import os
import json
from datetime import datetime

# Import your existing project components
from TransformerModel import DynamicTransformerModel, MissionPhase, MissionMemorySystem
from networking import NetworkCommunication, CommandHandler, CommandParser
from sensor import EnhancedSensorData
from AIOptimser import AIOptimiser
from astroDataManager import AstroDataManager
from optimiser import FuelOptimiser
from MissionPhaseAwareness import MissionPhaseAwareness
from AnomalyDetector import AnomalyDetector, EnvironmentalHazardDetector, EnsembleAnomalyDetector
from UnitTest.mission_evaluator import MarsDeepSpaceEvaluator
# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MarsMission")
logger.setLevel(logging.INFO)


class MarsExplorationMission:
    """
    Controls and manages a Mars exploration mission, integrating spacecraft
    movement, sensor data, and mission phases.
    """

    def __init__(self, mission_name="Mars Pathfinder II", mission_duration=100):
        self.mission_name = mission_name
        self.mission_duration = mission_duration
        self.mission_step = 0
        self.mission_complete = False
        sensor_config = {
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
            },
            "measurement_ranges": {
                "temperature": [-270, 500],  # Celsius
                "radiation": [0, 1000],  # mSv/h
                "magnetic": [0, 2000]  # nT
            }
        }

        # Initialize components
        self.astro_data = AstroDataManager()
        self.sensor_data = EnhancedSensorData(sensor_config)
        self.command_handler = CommandHandler()
        self.phase_awareness = MissionPhaseAwareness()
        self.fuel_optimizer = FuelOptimiser(self.astro_data)
        self.ai_optimizer = AIOptimiser(None)

        self.memory_system = MissionMemorySystem(max_episodes=200)

        # Command parser for mission directives
        self.command_parser = CommandParser()

        # Available actions
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

        # Initialize hazard detection system
        self.hazard_detector = EnsembleAnomalyDetector()
        # Calculate input dimension based on sensor data and state representation
        input_dim = 12  # Position (3), Velocity (3), Target position (3), Fuel (1), Time (1), Phase encoding (1)
        num_actions = len(self.actions)  # Number of available actions

        # Initialize a Transformer model for decision-making
        self.transformer_model = DynamicTransformerModel(input_dim=input_dim, num_actions=num_actions)
        self.mission_evaluator = MarsDeepSpaceEvaluator(max_steps=mission_duration, evaluation_frequency=25)

        # Mission history
        self.action_history = []
        self.reward_history = []
        self.phase_history = []
        self.position_history = []
        self.position_history = []
        self.fuel_level_history = []

        # Initialize mission parameters
        self.initialize_mission()

    def initialize_mission(self):
        """Set up the initial mission parameters and environment."""
        # Set up Mars as a mission target
        self.astro_data.add_celestial_object("Mars", {
            "position": np.array([140e9, 0, 0]),  # 140 million km from origin
            "mass": 6.39e23,  # kg
            "radius": 3389.5e3,  # m
            "atmosphere": True,
            "landing_sites": [
                {"name": "Jezero Crater", "position": np.array([140e9, 100e3, 50e3])},
                {"name": "Olympus Mons", "position": np.array([140e9, -200e3, 100e3])},
                {"name": "Valles Marineris", "position": np.array([140e9, 150e3, -100e3])}
            ]
        })

        # Add mission-specific targets on Mars
        self.mission_targets = [
            {"name": "Ancient Riverbed", "position": np.array([140e9, 120e3, 80e3]), "science_value": 0.8},
            {"name": "Subsurface Ice", "position": np.array([140e9, -150e3, 50e3]), "science_value": 0.9},
            {"name": "Magnetic Anomaly", "position": np.array([140e9, 100e3, -120e3]), "science_value": 0.7}
        ]

        # Set up Earth as starting point
        self.astro_data.add_celestial_object("Earth", {
            "position": np.array([0, 0, 0]),
            "mass": 5.97e24,
            "radius": 6371e3,
            "atmosphere": True
        })

        # Initialize spacecraft position, velocity, and target
        earth_pos = self.astro_data.get_object_data("Earth")["position"]
        mars_pos = self.astro_data.get_object_data("Mars")["position"]

        # Starting from Earth, heading to Mars
        self.spacecraft_position = earth_pos.copy() + np.array([1e9, 0, 0])  # 1 million km from Earth
        self.spacecraft_velocity = np.array([50e3, 0, 0])  # 50 km/s initial velocity
        self.target_position = mars_pos.copy()
        self.target_waypoints = self.generate_waypoints(earth_pos, mars_pos, 5)
        self.current_waypoint_index = 0
        self.current_waypoint = self.target_waypoints[0]

        # Initial spacecraft status
        self.fuel_level = 1.0
        self.mission_phase = "LAUNCH"
        self.anomaly_detected = False
        self.current_samples = []

        # Update sensor data with initial readings
        self.update_sensor_readings()

        # Initialize the anomaly detector ensemble
        self.input_dim = 10  # Based on the dimension of preprocessed sensor data
        self.anomaly_detector = EnsembleAnomalyDetector(voting_threshold=0.5)

        # Add standard anomaly detector
        standard_detector = AnomalyDetector(input_dim=self.input_dim)
        self.anomaly_detector.add_detector(standard_detector)

        # Add environmental hazard detector
        self.hazard_detector = EnvironmentalHazardDetector(input_dim=self.input_dim)
        self.anomaly_detector.add_detector(self.hazard_detector)

        # Add diverse detector for better ensemble performance
        diverse_detector = AnomalyDetector(input_dim=self.input_dim, latent_dim=6, threshold_multiplier=2.8)
        self.anomaly_detector.add_detector(diverse_detector)

        # Try to load pre-trained detector models
        try:
            if os.path.exists('hazard_detector_model_threshold.json'):
                with open('hazard_detector_model_threshold.json', 'r') as f:
                    threshold_data = json.load(f)
                    self.hazard_detector.anomaly_threshold = threshold_data.get('anomaly_threshold', 0.7)
                    logger.info(f"Loaded hazard detector threshold: {self.hazard_detector.anomaly_threshold}")
        except Exception as e:
            logger.warning(f"Could not load detector models: {e}")

        # Initialize the transformer model for decision making
        self.num_actions = 8  # Number of possible actions
        self.model = DynamicTransformerModel(self.input_dim, self.num_actions, max_sequence_length=25)

        # Initialize experience buffer for training
        self.experience_buffer = []
        self.normal_data_buffer = []  # For anomaly detector training

        # Model training parameters
        self.training_frequency = 10  # Train every 10 steps
        self.batch_size = 64  # Batch size for training
        self.anomaly_detector_trained = False
        self.hazard_detector_trained = False

        # Initialize state history for sequential models
        self.state_history = []
        for _ in range(self.model.max_sequence_length):
            self.state_history.append(np.zeros(self.input_dim))

        logger.info(f"Initialized {self.mission_name}")
        logger.info(f"Initial position: {np.linalg.norm(self.spacecraft_position) / 1e9:.2f} million km from origin")
        logger.info(f"Target: Mars at {np.linalg.norm(mars_pos) / 1e9:.2f} million km")
        logger.info(f"Mission duration: {self.mission_duration} steps")



    def generate_waypoints(self, start_pos, end_pos, num_waypoints):
        """Generate a series of waypoints between start and end positions."""
        waypoints = []
        for i in range(num_waypoints):
            # Linear interpolation with some randomness for realistic waypoints
            t = (i + 1) / (num_waypoints + 1)
            waypoint = start_pos + t * (end_pos - start_pos)

            # Add some variability to avoid straight-line path
            if i > 0 and i < num_waypoints - 1:
                # Add random offset perpendicular to trajectory
                trajectory_dir = (end_pos - start_pos) / np.linalg.norm(end_pos - start_pos)
                perp_dir1 = np.array([-trajectory_dir[1], trajectory_dir[0], 0])
                perp_dir2 = np.cross(trajectory_dir, perp_dir1)

                # Random offset magnitude (up to 5% of the total distance)
                max_offset = 0.05 * np.linalg.norm(end_pos - start_pos)
                random_offset = (random.random() * 2 - 1) * max_offset * perp_dir1 + \
                                (random.random() * 2 - 1) * max_offset * perp_dir2

                waypoint += random_offset

            waypoints.append(waypoint)

        return waypoints

    def train_anomaly_detector(self, force=False):
        """Train the anomaly detector on collected normal data."""
        if (len(self.normal_data_buffer) >= 200 and not self.anomaly_detector_trained) or force:
            normal_data = np.array(self.normal_data_buffer)
            logger.info(f"Training anomaly detector ensemble on {len(normal_data)} samples")

            # Train each detector in the ensemble
            for i, detector in enumerate(self.anomaly_detector.detectors):
                if isinstance(detector, EnvironmentalHazardDetector):
                    # Create sample hazard labels for different environmental aspects
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
            self.hazard_detector_trained = True

            # Save detector thresholds
            for i, detector in enumerate(self.anomaly_detector.detectors):
                if isinstance(detector, EnvironmentalHazardDetector):
                    threshold_data = {
                        "anomaly_threshold": detector.anomaly_threshold
                    }

                    # Only add these attributes if they exist
                    if hasattr(detector, 'mean_reconstruction_error'):
                        threshold_data["mean_error"] = detector.mean_reconstruction_error
                    if hasattr(detector, 'std_reconstruction_error'):
                        threshold_data["std_error"] = detector.std_reconstruction_error

                    with open('hazard_detector_model_threshold.json', 'w') as f:
                        json.dump(threshold_data, f)

            logger.info("Anomaly detector ensemble trained successfully")
            return True
        return False

    def detect_anomalies_ml(self):
        """Use machine learning to detect anomalies instead of random generation."""
        if not self.anomaly_detector_trained:
            # Fall back to random detection if not trained
            return self.detect_anomalies()

        # Get current state
        current_features = self.preprocess_sensor_data()

        # Use ensemble to detect anomalies - only unpack the values actually returned
        is_anomaly, confidence = self.anomaly_detector.detect_anomaly(
            np.array([current_features]),
            self.mission_step
        )

        # Check for environmental hazards
        for detector in self.anomaly_detector.detectors:
            if isinstance(detector, EnvironmentalHazardDetector):
                is_hazard, hazard_score, hazard_type = detector.detect_hazard(
                    np.array([current_features]),
                    mission_step=self.mission_step
                )

                if is_hazard:
                    logger.warning(f"Environmental hazard detected: {hazard_type} (score: {hazard_score:.4f})")
                    self.sensor_data.current_readings['environmental_hazard'] = True
                    self.sensor_data.current_readings['hazard_type'] = hazard_type
                    self.sensor_data.current_readings['hazard_score'] = hazard_score

        # Update anomaly status
        self.anomaly_detected = is_anomaly
        self.sensor_data.current_readings['anomaly_detector'] = is_anomaly
        self.sensor_data.current_readings['anomaly_score'] = confidence

        if is_anomaly:
            logger.warning(f"Anomaly detected by ML system with confidence: {confidence:.4f}")

        return is_anomaly

    def train_model(self):
        """Train the transformer model on collected experiences."""
        # Skip if we don't have enough samples yet
        if len(self.experience_buffer) < self.batch_size:
            logger.info(f"Not enough experiences for training: {len(self.experience_buffer)}/{self.batch_size}")
            return False

        # Sample a batch of experiences
        batch_indices = np.random.choice(len(self.experience_buffer), self.batch_size, replace=True)
        batch = [self.experience_buffer[i] for i in batch_indices]

        # Extract components from experiences
        state_sequences = []
        targets = []

        for state, action, reward, next_state, done in batch:
            # Create state sequence (current state with history)
            state_sequence = np.array(list(self.state_history)[-(self.model.max_sequence_length - 1):] + [state])

            # Prepare next state sequence
            next_state_sequence = np.array(
                list(self.state_history)[-(self.model.max_sequence_length - 1) + 1:] + [next_state])

            # Get current model's Q-value predictions
            target = self.model.predict(state_sequence)[0]

            # Ensure target array has the correct length
            if len(target) != self.num_actions:
                logger.warning(f"Target length mismatch: {len(target)} vs {self.num_actions}")
                # Pad or truncate target to match num_actions
                if len(target) < self.num_actions:
                    target = np.pad(target, (0, self.num_actions - len(target)), 'constant')
                else:
                    target = target[:self.num_actions]

            # Calculate updated Q-value for the action taken
            if done:
                target[action] = reward
            else:
                next_q_values = self.model.predict(next_state_sequence)[0]
                # Ensure next_q_values has correct dimensions
                if len(next_q_values) != self.num_actions:
                    next_q_values = next_q_values[:self.num_actions] if len(
                        next_q_values) > self.num_actions else np.pad(
                        next_q_values, (0, self.num_actions - len(next_q_values)), 'constant')
                target[action] = reward + 0.95 * np.max(next_q_values)  # 0.95 is gamma discount factor

            state_sequences.append(state_sequence)
            targets.append(target)

        # Convert lists to numpy arrays before training
        state_sequences = np.array(state_sequences)
        targets = np.array(targets)

        # Train the model
        self.model.train(state_sequences, targets)
        logger.info(f"Trained model on batch of {len(targets)} experiences")

        # Save the model after successful training
        self.save_trained_components()
        logger.info("Saved model after training")

        return True

    def update_sensor_readings(self):
        """Update the sensor data with current spacecraft state."""
        self.sensor_data.current_readings = {
            "position": self.spacecraft_position,
            "velocity": self.spacecraft_velocity,
            "target_position": self.current_waypoint,
            "fuel_level": self.fuel_level,
            "mission_time": self.mission_step,
            "mission_phase": self.mission_phase,
            "anomaly_detected": self.anomaly_detected,
            "collected_samples": len(self.current_samples),
            "distance_to_target": np.linalg.norm(self.current_waypoint - self.spacecraft_position)
        }

        # Add some sensor noise for realism
        for key in ["position", "velocity"]:
            if key in self.sensor_data.current_readings:
                value = self.sensor_data.current_readings[key]
                if isinstance(value, np.ndarray) and value.size > 0:
                    noise = np.random.normal(0, 0.01 * np.linalg.norm(value), value.shape)
                    self.sensor_data.current_readings[key] = value + noise


    def determine_mission_phase(self):
        """Determine the current mission phase based on spacecraft state."""
        # Get current state values
        position = self.spacecraft_position
        velocity = self.spacecraft_velocity
        fuel = self.fuel_level
        mission_time = self.mission_step

        # Distance to Earth and Mars
        earth_pos = self.astro_data.get_object_data("Earth")["position"]
        mars_pos = self.astro_data.get_object_data("Mars")["position"]
        dist_to_earth = np.linalg.norm(position - earth_pos)
        dist_to_mars = np.linalg.norm(position - mars_pos)

        # LAUNCH: Early mission and close to Earth
        if mission_time < self.mission_duration * 0.15 and dist_to_earth < 20e9:
            return "LAUNCH"

        # CRITICAL: Low fuel or anomaly detected
        if fuel < 0.25 or self.anomaly_detected:
            return "CRITICAL"

        # EMERGENCY: Very low fuel or severe anomaly
        if fuel < 0.15 or (self.anomaly_detected and fuel < 0.3):
            return "EMERGENCY"

        # EXPLORATION: Near Mars and low velocity
        if dist_to_mars < 10e9 and np.linalg.norm(velocity) < 30e3:
            return "EXPLORATION"

        # Default is TRANSIT
        return "TRANSIT"

    def preprocess_sensor_data(self):
        """Preprocess sensor data for input to the model and anomaly detectors."""
        readings = self.sensor_data.current_readings

        # Extract and normalize features
        position = readings['position'] / 1e11  # Normalize position
        velocity = readings['velocity'] / 1e5  # Normalize velocity
        fuel = readings['fuel_level']  # Already normalized 0-1

        # Create feature vector
        features = np.concatenate([
            position,  # 3 values
            velocity,  # 3 values
            [fuel],  # 1 value
            [float(self.anomaly_detected)],  # 1 value
            [self.mission_step / self.mission_duration],  # 1 value
            [np.linalg.norm(self.spacecraft_position - self.current_waypoint) / 1e11]  # 1 value - distance to waypoint
        ])

        return features



    def detect_anomalies(self):
        """Randomly generate potential anomalies during the mission."""
        # Base probability of anomaly
        base_prob = 0.03  # 3% chance per step

        # Adjust probability based on mission phase
        phase_multipliers = {
            "LAUNCH": 1.5,  # Increased risk during launch
            "TRANSIT": 1.0,  # Base risk during transit
            "EXPLORATION": 1.2,  # Slightly increased during exploration
            "CRITICAL": 0.8,  # Lower as we're already in critical mode
            "EMERGENCY": 0.5  # Lowest as we're already in emergency
        }

        # Calculate final probability
        anomaly_probability = base_prob * phase_multipliers.get(self.mission_phase, 1.0)

        # Generate random anomaly
        if random.random() < anomaly_probability and not self.anomaly_detected:
            self.anomaly_detected = True
            anomaly_types = [
                "Radiation spike detected",
                "Micrometeoroid impact on shield",
                "Communication system fluctuation",
                "Power system irregularity",
                "Navigation sensor glitch"
            ]
            anomaly = random.choice(anomaly_types)
            logger.warning(f"ANOMALY DETECTED: {anomaly}")
            return True

        # Chance to clear existing anomaly
        elif self.anomaly_detected and random.random() < 0.2:
            self.anomaly_detected = False
            logger.info("Anomaly cleared")

        return self.anomaly_detected

    def update_position(self, action_name):
        """Update spacecraft position based on the action taken."""
        # Time step (simulation time that passes in one mission step)
        time_step = 3600 * 4  # 4 hours in seconds

        # Save current values for fuel calculations
        original_velocity = self.spacecraft_velocity.copy()
        original_position = self.spacecraft_position.copy()

        # Get direction to current waypoint
        to_waypoint = self.current_waypoint - self.spacecraft_position
        distance_to_waypoint = np.linalg.norm(to_waypoint)

        if distance_to_waypoint > 0:
            direction_to_waypoint = to_waypoint / distance_to_waypoint
        else:
            # If at waypoint, use current velocity direction or default forward
            if np.any(self.spacecraft_velocity):
                direction_to_waypoint = self.spacecraft_velocity / np.linalg.norm(self.spacecraft_velocity)
            else:
                direction_to_waypoint = np.array([1.0, 0, 0])
                return None
            return None

            # Maximum velocity cap (km/s)
        MAX_VELOCITY = 40e3  # 40,000 km/s cap

        # Apply action effects
        if action_name == "maintain_course":
            # Keep current velocity
            pass

        elif action_name == "increase_velocity":
            # Accelerate in the direction toward the waypoint
            acceleration = 100  # 100 m/s² (much more reasonable)
            delta_v = direction_to_waypoint * acceleration * time_step
            new_velocity = self.spacecraft_velocity + delta_v

            # Cap velocity if it exceeds maximum
            new_speed = np.linalg.norm(new_velocity)
            if new_speed > MAX_VELOCITY:
                new_velocity = new_velocity * (MAX_VELOCITY / new_speed)

            self.spacecraft_velocity = new_velocity

        elif action_name == "decrease_velocity":
            # Decelerate current velocity
            if np.any(self.spacecraft_velocity):
                current_speed = np.linalg.norm(self.spacecraft_velocity)
                deceleration = min(100 * time_step, current_speed * 0.3)  # Reduce by 30% or by deceleration parameter
                self.spacecraft_velocity -= (self.spacecraft_velocity / current_speed) * deceleration

        elif action_name == "adjust_trajectory":
                # Change direction toward waypoint while maintaining speed
                if np.any(self.spacecraft_velocity):
                    current_speed = np.linalg.norm(self.spacecraft_velocity)
                    # Mix current direction with target direction (gradual turn)
                    current_direction = self.spacecraft_velocity / current_speed
                    new_direction = 0.7 * direction_to_waypoint + 0.3 * current_direction
                    new_direction = new_direction / np.linalg.norm(new_direction)
                    self.spacecraft_velocity = new_direction * current_speed

        elif action_name == "investigate_anomaly":
            # Slow down for investigation
            if np.any(self.spacecraft_velocity):
                current_speed = np.linalg.norm(self.spacecraft_velocity)
                self.spacecraft_velocity *= 0.6  # Reduce speed by 40%

            # Chance to collect valuable samples during investigation
            if random.random() < 0.4 and self.mission_phase == "EXPLORATION":
                sample_value = random.uniform(0.6, 1.0)
                self.current_samples.append({"value": sample_value, "step": self.mission_step})
                logger.info(f"Sample collected! Scientific value: {sample_value:.2f}")

        elif action_name == "refuel":
            # Refuel the spacecraft (in a real simulation, this would be based on proximity to resources)
            refuel_amount = 0.2  # Refuel by 20%
            self.fuel_level = min(1.0, self.fuel_level + refuel_amount)
            logger.info(f"Refueled spacecraft. New fuel level: {self.fuel_level:.2f}")

        elif action_name == "emergency_protocol":
            # Emergency protocol - reduce velocity significantly and stabilize
            if np.any(self.spacecraft_velocity):
                self.spacecraft_velocity *= 0.3  # Reduce speed by 70%

            # Chance to clear anomalies
            if self.anomaly_detected and random.random() < 0.6:
                self.anomaly_detected = False
                logger.info("Emergency protocol successful - anomaly cleared")

        elif action_name == "return_to_base":
            # Change direction toward Earth if in emergency
            if self.mission_phase in ["EMERGENCY", "CRITICAL"]:
                earth_pos = self.astro_data.get_object_data("Earth")["position"]
                to_earth = earth_pos - self.spacecraft_position
                if np.any(to_earth):
                    direction_to_earth = to_earth / np.linalg.norm(to_earth)

                    # Keep current speed but change direction
                    if np.any(self.spacecraft_velocity):
                        current_speed = np.linalg.norm(self.spacecraft_velocity)
                        new_direction = 0.8 * direction_to_earth + 0.2 * (self.spacecraft_velocity / current_speed)
                        new_direction = new_direction / np.linalg.norm(new_direction)
                        self.spacecraft_velocity = new_direction * current_speed

        # Update position using velocity
        self.spacecraft_position += self.spacecraft_velocity * time_step

        # Calculate fuel consumption based on action and velocity change
        velocity_change = np.linalg.norm(self.spacecraft_velocity - original_velocity)

        # Base fuel consumption rates per phase
        base_consumption = self.fuel_optimizer.consumption_rates.get(self.mission_phase, 0.01)  # Reduced from 0.02

        # Additional consumption for acceleration/deceleration
        action_consumption = {
            "maintain_course": 0,
            "adjust_trajectory": 0.002,  # Reduced from 0.005
            "increase_velocity": 0.005,  # Reduced from 0.01
            "decrease_velocity": 0.003,  # Reduced from 0.007
            "investigate_anomaly": 0.004,  # Reduced from 0.006
            "refuel": -0.25,  # Improved refuel effect from -0.18
            "emergency_protocol": 0.007,  # Reduced from 0.015
            "return_to_base": 0.004  # Reduced from 0.008
        }

        # Calculate total fuel consumption
        fuel_consumed = base_consumption + action_consumption.get(action_name, 0)

        # Add velocity-based consumption with a much lower scale factor
        current_velocity_magnitude = np.linalg.norm(self.spacecraft_velocity) / 1e3  # km/s
        velocity_change = np.linalg.norm(self.spacecraft_velocity - original_velocity) / 1e3  # km/s

        # Scale fuel consumption based on current velocity and change
        velocity_factor = 0.00001 * min(velocity_change, 1.0)  # Greatly reduced factor
        fuel_consumed += velocity_factor

        # Ensure total consumption is reasonable
        fuel_consumed = min(fuel_consumed, 0.05)  # Maximum 5% consumption per step (was 10%)

        # Apply the fuel consumption
        self.fuel_level = max(0, self.fuel_level - fuel_consumed)
        # Check waypoint progress
        if distance_to_waypoint < 5e9:  # Within 5 million km of waypoint
            if self.current_waypoint_index < len(self.target_waypoints) - 1:
                self.current_waypoint_index += 1
                self.current_waypoint = self.target_waypoints[self.current_waypoint_index]
                logger.info(f"Reached waypoint! Moving to next waypoint")

    def evaluate_action(self, action_name):
        """Evaluate the action taken and return a reward value."""
        # Basic reward setting
        reward = 0.0

        # Current state metrics
        distance_to_waypoint = np.linalg.norm(self.current_waypoint - self.spacecraft_position)
        velocity_magnitude = np.linalg.norm(self.spacecraft_velocity)

        # Get previous position from history (if available)
        if self.position_history:
            prev_position = self.position_history[-1]
            prev_distance = np.linalg.norm(self.current_waypoint - prev_position)

            # Reward for approaching the waypoint
            if distance_to_waypoint < prev_distance:
                reward += 0.2
            else:
                reward -= 0.1

        # Phase-specific rewards
        if self.mission_phase == "LAUNCH":
            # During launch, we want to accelerate quickly toward Mars
            if action_name == "increase_velocity":
                reward += 0.3
            elif action_name == "adjust_trajectory":
                reward += 0.2

        elif self.mission_phase == "TRANSIT":
            # During transit, we want to maintain efficient course toward Mars
            if action_name == "maintain_course" and not self.anomaly_detected:
                reward += 0.2
            elif action_name == "adjust_trajectory":
                reward += 0.1

        elif self.mission_phase == "EXPLORATION":
            # During exploration, we want to investigate and collect samples
            if action_name == "investigate_anomaly":
                reward += 0.4
            elif action_name == "adjust_trajectory":
                reward += 0.2

        elif self.mission_phase == "CRITICAL":
            # During critical situations, we prioritize safety
            if action_name == "emergency_protocol":
                reward += 0.3
            elif self.anomaly_detected and action_name == "investigate_anomaly":
                reward += 0.2
            elif self.fuel_level < 0.3 and action_name == "refuel":
                reward += 0.4

        elif self.mission_phase == "EMERGENCY":
            # During emergency, focus on returning or stabilizing
            if action_name == "emergency_protocol":
                reward += 0.5
            elif action_name == "return_to_base":
                reward += 0.4
            elif self.fuel_level < 0.2 and action_name == "refuel":
                reward += 0.6

        # Anomaly handling rewards
        if self.anomaly_detected:
            if action_name == "investigate_anomaly":
                reward += 0.3
            elif action_name == "emergency_protocol":
                reward += 0.2

        # Fuel efficiency rewards
        if self.fuel_level < 0.3:
            if action_name == "refuel":
                reward += 0.3
            elif action_name in ["increase_velocity", "emergency_protocol"] and self.fuel_level < 0.2:
                reward -= 0.2  # Penalize fuel-intensive actions when low on fuel

        # Waypoint achievement reward
        if distance_to_waypoint < 5e9 and self.current_waypoint_index > 0:
            reward += 0.5

        # Mission completion bonus
        mars_pos = self.astro_data.get_object_data("Mars")["position"]
        dist_to_mars = np.linalg.norm(self.spacecraft_position - mars_pos)
        if dist_to_mars < 5e9 and self.mission_step > self.mission_duration * 0.7:
            reward += 1.0
            if not self.mission_complete:
                logger.info("MISSION SUCCESSFUL! Reached Mars vicinity!")
                self.mission_complete = True

        return reward

    def _select_action_with_memory(self, current_state, similar_episodes, context_episodes, location_episodes):
        """Use episodic and semantic memory to enhance action selection."""
        # Default to model-based action selection
        if self.model and hasattr(self.model, 'trained') and self.model.trained:
            # Get state sequence for model input
            state_sequence = np.array(self.state_history)
            # Get Q-values from model
            q_values = self.model.predict(state_sequence)[0]

            # Modify Q-values based on memory
            modified_q_values = q_values.copy()

            # Boost actions that were successful in similar states
            if similar_episodes:
                for episode, similarity in similar_episodes:
                    action_id = self.action_to_id.get(episode["action"], 0)
                    if action_id < len(modified_q_values):
                        reward_factor = min(max(episode["reward"], -2), 2) / 2  # Normalize to [-1, 1]
                        boost = similarity * reward_factor * 0.3  # Scale by similarity and reward
                        modified_q_values[action_id] += boost

            # Boost actions that worked well in the current phase
            if context_episodes:
                action_rewards = {}
                for episode in context_episodes:
                    action = episode["action"]
                    if action not in action_rewards:
                        action_rewards[action] = {"sum": 0, "count": 0}
                    action_rewards[action]["sum"] += episode["reward"]
                    action_rewards[action]["count"] += 1

                for action, stats in action_rewards.items():
                    if stats["count"] > 0:
                        avg_reward = stats["sum"] / stats["count"]
                        action_id = self.action_to_id.get(action, 0)
                        if action_id < len(modified_q_values):
                            boost = max(min(avg_reward, 2), -2) * 0.2  # Scale appropriately
                            modified_q_values[action_id] += boost

            # Consider actions taken in similar locations
            if location_episodes:
                location_action_counts = {}
                for episode in location_episodes:
                    action = episode["action"]
                    if action not in location_action_counts:
                        location_action_counts[action] = {"sum": 0, "count": 0}
                    location_action_counts[action]["sum"] += episode["reward"]
                    location_action_counts[action]["count"] += 1

                for action, stats in location_action_counts.items():
                    if stats["count"] > 0:
                        avg_reward = stats["sum"] / stats["count"]
                        if avg_reward > 0:  # Only boost positive experiences
                            action_id = self.action_to_id.get(action, 0)
                            if action_id < len(modified_q_values):
                                boost = min(avg_reward, 1) * 0.15
                                modified_q_values[action_id] += boost

            # Use semantic knowledge in special situations
            if self.anomaly_detected:
                hazard_knowledge = self.memory_system.get_semantic_knowledge("hazards")
                if hazard_knowledge:
                    best_response = None
                    best_reward = -float('inf')

                    for key, data in hazard_knowledge.items():
                        if key.startswith(f"response_") and self.mission_phase in key:
                            avg_reward = data["total_reward"] / max(data["count"], 1)
                            if avg_reward > best_reward:
                                best_reward = avg_reward
                                best_response = key.split("_")[1]  # Extract action name

                    if best_response:
                        action_id = self.action_to_id.get(best_response, 0)
                        if action_id < len(modified_q_values):
                            modified_q_values[action_id] += 0.5  # Significant boost for proven hazard responses

            # Check for efficiency knowledge when fuel is low
            if self.fuel_level < 0.3:
                efficiency_knowledge = self.memory_system.get_semantic_knowledge("efficiency")
                if efficiency_knowledge:
                    fuel_bracket = int(self.fuel_level * 10)
                    best_action = None
                    best_efficiency = -float('inf')

                    for key, data in efficiency_knowledge.items():
                        if f"_fuel_{fuel_bracket}" in key:
                            avg_reward = data["total_reward"] / max(data["count"], 1)
                            if avg_reward > best_efficiency:
                                best_efficiency = avg_reward
                                best_action = key.split("_")[0]  # Extract action name

                    if best_action:
                        action_id = self.action_to_id.get(best_action, 0)
                        if action_id < len(modified_q_values):
                            modified_q_values[action_id] += 0.4  # Boost for fuel-efficient actions

            # Select action with highest modified Q-value
            action_id = np.argmax(modified_q_values)
            logger.info(f"Memory-enhanced model selected action with Q-values: {modified_q_values}")

            # Map action ID to action name
            action_name = self.actions.get(action_id, "maintain_course")
        else:
            # If model isn't trained, use memory-based heuristics
            if context_episodes:
                # Find most successful actions in similar contexts
                action_rewards = {}
                for episode in context_episodes:
                    action = episode["action"]
                    if action not in action_rewards:
                        action_rewards[action] = {"sum": 0, "count": 0}
                    action_rewards[action]["sum"] += episode["reward"]
                    action_rewards[action]["count"] += 1

                best_actions = []
                for action, stats in action_rewards.items():
                    if stats["count"] > 0:
                        avg_reward = stats["sum"] / stats["count"]
                        if avg_reward > 0:
                            best_actions.append((action, avg_reward))

                if best_actions:
                    best_actions.sort(key=lambda x: x[1], reverse=True)
                    # Weighted random choice based on rewards
                    weights = [max(0.1, r) for _, r in best_actions]
                    total = sum(weights)
                    weights = [w / total for w in weights]
                    return np.random.choice([a for a, _ in best_actions], p=weights)

            # Fall back to standard heuristic selection
            if self.anomaly_detected and random.random() < 0.7:
                action_name = "investigate_anomaly" if random.random() < 0.6 else "emergency_protocol"
            elif self.fuel_level < 0.2 and random.random() < 0.8:
                action_name = "refuel"
            else:
                actions = ["maintain_course", "adjust_trajectory", "increase_velocity", "decrease_velocity"]
                action_name = random.choice(actions)

        return action_name

    def run_mission_step(self):
        """Execute a single step of the Mars mission with memory system integration."""
        self.mission_step += 1
        logger.info(f"----- Mission Step {self.mission_step}/{self.mission_duration} -----")

        # Update mission phase based on current state
        old_phase = self.mission_phase
        self.mission_phase = self.determine_mission_phase()
        if old_phase != self.mission_phase:
            logger.info(f"Phase change: {old_phase} -> {self.mission_phase}")

        # Update spacecraft parameters and sensor readings
        self.update_sensor_readings()

        # Use ML for anomaly detection
        if self.anomaly_detector_trained:
            self.detect_anomalies_ml()
        else:
            self.detect_anomalies()  # Use random detection as fallback

            # Collect normal data for training when no anomalies detected
            if not self.anomaly_detected and self.mission_phase in ["TRANSIT", "EXPLORATION"]:
                current_features = self.preprocess_sensor_data()
                self.normal_data_buffer.append(current_features)

            # Try to train the detector if we have enough data
            if len(self.normal_data_buffer) >= 200:
                self.train_anomaly_detector()

        # Get current position and distance to waypoint
        dist_to_waypoint = np.linalg.norm(self.current_waypoint - self.spacecraft_position)
        logger.info(f"Distance to waypoint: {dist_to_waypoint / 1e9:.2f} million km")

        # Preprocess current state for model input
        current_state = self.preprocess_sensor_data()

        # Record in state history
        self.state_history.append(current_state)
        if len(self.state_history) > self.model.max_sequence_length:
            self.state_history.pop(0)  # Remove oldest state

        # Before action selection, retrieve relevant memories
        similar_episodes = self.memory_system.retrieve_by_similarity(current_state, top_k=3)
        context_episodes = self.memory_system.retrieve_by_context(
            anomaly=self.anomaly_detected,
            phase=self.mission_phase
        )
        location_episodes = self.memory_system.retrieve_by_location(
            self.spacecraft_position,
            radius=1e11  # 100 million km radius
        )

        # Use memories to influence action selection
        action_name = self._select_action_with_memory(current_state, similar_episodes,
                                                      context_episodes, location_episodes)

        logger.info(f"Selected action: {action_name} in phase {self.mission_phase}")

        # Update position based on the chosen action
        self.update_position(action_name)

        # Evaluate the action
        reward = self.evaluate_action(action_name)
        logger.info(f"Action resulted in reward: {reward:.4f}")

        # Store this episode in memory
        self.memory_system.store_episode(
            state=current_state,
            action=action_name,
            reward=reward,
            phase=self.mission_phase,
            position=self.spacecraft_position.copy(),
            anomaly_detected=self.anomaly_detected,
            timestamp=self.mission_step
        )

        # Update histories
        self.action_history.append(action_name)
        self.reward_history.append(reward)
        self.phase_history.append(self.mission_phase)
        self.position_history.append(self.spacecraft_position.copy())

        # Update mission evaluator with current data
        self.mission_evaluator.update_metrics(
            self,  # captain
            self.spacecraft_position.copy(),  # current position for trajectory
            self.fuel_level,  # current fuel level
            action_name,  # action taken this step
            self.mission_phase  # current phase
        )

        # Update memory utilization metric
        memory_utilization = self.memory_system.calculate_memory_utilization()
        if hasattr(self.sensor_data, 'current_readings'):
            self.sensor_data.current_readings['memory_utilization'] = memory_utilization

        # Check if mission is complete
        if self.mission_step >= self.mission_duration:
            self.mission_complete = True
            logger.info("Mission duration reached - mission complete")

        return action_name, reward

    def save_trained_components(self):
        """Save trained models to disk."""
        try:
            # Use the save_model method instead of save_weights
            self.model.save_model(f"models/mars_mission_{self.mission_name}")


            # Save anomaly detector if it exists and has been trained
            if hasattr(self, 'anomaly_detector') and hasattr(self.anomaly_detector, 'save_ensemble'):
                self.anomaly_detector.save_ensemble(f"models/anomaly_detector_{self.mission_name}")

            # Save hazard detector if it exists
            if hasattr(self, 'hazard_detector') and hasattr(self.hazard_detector, 'save_model'):
                self.hazard_detector.save_model(f"models/hazard_detector_{self.mission_name}")

            # Save mission metadata
            metadata = {
                "mission_name": self.mission_name,
                "last_saved": datetime.now().isoformat(),
                "mission_step": self.mission_step,
                "epsilon": self.epsilon
            }

            os.makedirs("models", exist_ok=True)
            with open(f"models/mission_metadata_{self.mission_name}.json", "w") as f:
                json.dump(metadata, f)

            logger.info(f"Saved trained components for mission {self.mission_name}")
            return True
        except Exception as e:
            logger.error(f"Error saving trained components: {str(e)}")
            return False

    def run_full_mission(self):
        """Run the complete Mars mission until completion."""
        logger.info(f"Starting Mars mission: {self.mission_name}")

        running = True
        while running and self.mission_step < self.mission_duration:
            running = self.run_mission_step()
            time.sleep(0.1)  # Small delay for visual monitoring

            # Add periodic saving during the mission
            if self.mission_step % 50 == 0:  # Save every 50 steps
                self.save_trained_components()
                logger.info(f"Saved model checkpoint at step {self.mission_step}")

        # Ensure final save happens regardless of how mission ends
        self.save_trained_components()
        logger.info("Saved final model state at mission end")

        # Generate mission report
        self.generate_mission_report()

    def generate_mission_report(self):
        """Generate a detailed mission report with memory analytics and mission evaluation."""
        logger.info(f"\n{'=' * 50}\nMISSION REPORT: {self.mission_name}\n{'=' * 50}")
        logger.info(f"Mission duration: {self.mission_step}/{self.mission_duration} steps")

        # Calculate mission success metrics
        total_reward = sum(self.reward_history)
        avg_reward = total_reward / max(1, len(self.reward_history))
        final_distance = np.linalg.norm(self.spacecraft_position - self.target_position)

        logger.info(f"Total mission reward: {total_reward:.2f}")
        logger.info(f"Average reward per step: {avg_reward:.4f}")
        logger.info(f"Final distance to target: {final_distance / 1e6:.2f} km")
        logger.info(f"Final fuel level: {self.fuel_level:.2%}")

        # Phase analysis
        phase_counts = {}
        for phase in self.phase_history:
            if phase not in phase_counts:
                phase_counts[phase] = 0
            phase_counts[phase] += 1

        logger.info("\nMission Phase Analysis:")
        for phase, count in phase_counts.items():
            percentage = (count / len(self.phase_history)) * 100
            logger.info(f"  {phase}: {count} steps ({percentage:.1f}%)")

        # Memory system analytics
        logger.info("\nMemory System Analytics:")
        logger.info(f"  Memory Utilization: {self.memory_system.memory_utilization:.2f}")
        logger.info(f"  Episodic Memories: {len(self.memory_system.episodic_memory)}")

        # Most important memories
        important_memories = sorted(
            self.memory_system.episodic_memory,
            key=lambda x: x["importance"],
            reverse=True
        )[:3]

        logger.info("  Top 3 Important Memories:")
        for i, mem in enumerate(important_memories):
            logger.info(f"    {i + 1}. Phase: {mem['phase']}, Action: {mem['action']}, Reward: {mem['reward']:.2f}")

        # Memory retrieval statistics
        most_retrieved = sorted(
            self.memory_system.episodic_memory,
            key=lambda x: x["retrieval_count"],
            reverse=True
        )[:3]

        logger.info("  Most Frequently Retrieved Memories:")
        for i, mem in enumerate(most_retrieved):
            logger.info(f"    {i + 1}. Retrieved {mem['retrieval_count']} times - Action: {mem['action']}")

        # Semantic memory insights
        hazard_knowledge = self.memory_system.get_semantic_knowledge("hazards")
        if hazard_knowledge:
            best_hazard_responses = sorted(
                [(k, v["total_reward"] / v["count"]) for k, v in hazard_knowledge.items() if v["count"] > 0],
                key=lambda x: x[1],
                reverse=True
            )[:2]

            if best_hazard_responses:
                logger.info("  Best Hazard Responses:")
                for response, avg_reward in best_hazard_responses:
                    logger.info(f"    {response}: Avg Reward {avg_reward:.2f}")

        # Mission pattern insights
        pattern_knowledge = self.memory_system.get_semantic_knowledge("mission_patterns")
        if pattern_knowledge:
            logger.info("  Key Mission Patterns:")
            for pattern, count in sorted(pattern_knowledge.items(), key=lambda x: x[1], reverse=True)[:3]:
                logger.info(f"    {pattern}: {count} occurrences")

        # Add mission evaluation report from MarsDeepSpaceEvaluator
        logger.info("\nDetailed Mission Evaluation:")
        eval_report = self.mission_evaluator.generate_evaluation_report()
        logger.info(eval_report)

        # Generate and display evaluation visualization
        try:
            eval_fig = self.mission_evaluator.plot_evaluation_metrics()
            logger.info("Mission evaluation visualization generated")
        except Exception as e:
            logger.error(f"Failed to generate evaluation visualization: {e}")

        logger.info(f"{'=' * 50}")
        return {
            "total_reward": total_reward,
            "avg_reward": avg_reward,
            "final_distance": final_distance,
            "fuel_level": self.fuel_level,
            "memory_utilization": self.memory_system.memory_utilization,
            "episodic_memories": len(self.memory_system.episodic_memory),
            "evaluation": self.mission_evaluator.evaluate_mission(
                self,
                np.array(self.position_history),
                self.reward_history,
                self.action_history,
                self.phase_history,
                self.target_waypoints
            )
        }

# Main execution function
def run_mars_exploration_mission(mission_name="Mars Pathfinder II", duration=450):
    """Run a complete Mars exploration mission."""
    mission = MarsExplorationMission(mission_name=mission_name, mission_duration=duration)
    mission.run_full_mission()
    return mission


if __name__ == "__main__":
    # Run a Mars exploration mission
    run_mars_exploration_mission()