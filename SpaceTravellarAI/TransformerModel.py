from datetime import time

import numpy as np
import logging
import json
import tensorflow as tf
from tensorflow.data.experimental import AutoShardPolicy
import os



class TransformerModel:
    """Transformer-based neural network for decision-making."""

    def __init__(self, input_dim, num_actions, sequence_length=50, parameter_support=True):
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.sequence_length = sequence_length
        self.parameter_support = parameter_support
        self.model = self._build_model()
        # Parameter prediction ranges for different actions
        self.parameter_ranges = {
            'adjust_trajectory': {
                'angle_x': (-45.0, 45.0),  # Degrees
                'angle_y': (-45.0, 45.0),  # Degrees
                'angle_z': (-45.0, 45.0),  # Degrees
                'magnitude': (0.1, 1.0)  # Relative thrust magnitude
            },
            'increase_velocity': {
                'delta_v': (0.1, 5.0)  # Speed increase in units/second
            },
            'decrease_velocity': {
                'delta_v': (0.1, 5.0)  # Speed decrease in units/second
            },
            'investigate_anomaly': {
                'scan_intensity': (1, 10),  # Level of scan detail
                'approach_distance': (10, 100)  # How close to get (meters)
            },
            'refuel': {
                'amount': (0.1, 1.0)  # Relative amount (percentage of max)
            }
        }

    def _build_model(self):
        """Build the Transformer model architecture with parameter prediction heads."""
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0") if tf.config.list_physical_devices(
            'GPU') else tf.distribute.OneDeviceStrategy(device="/cpu:0")

        with tf.distribute.get_strategy().scope():
            # Input shape: [batch_size, sequence_length, input_dim]
            inputs = tf.keras.layers.Input(shape=(self.sequence_length, self.input_dim))

            # Positional Encoding
            position = tf.range(start=0, limit=self.sequence_length, delta=1)
            position = tf.keras.layers.Embedding(input_dim=self.sequence_length, output_dim=self.input_dim)(position)
            x = inputs + position

            # Transformer Encoder with more layers and heads
            for _ in range(6):
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
                x = tf.keras.layers.MultiHeadAttention(key_dim=64, num_heads=8, dropout=0.1)(x, x)
                x = tf.keras.layers.Dropout(0.1)(x)
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
                ffn = tf.keras.Sequential([
                    tf.keras.layers.Dense(512, activation='relu'),
                    tf.keras.layers.Dropout(0.1),
                    tf.keras.layers.Dense(self.input_dim)
                ])
                x = ffn(x)
                x = tf.keras.layers.Add()([x, inputs])
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            shared_features = tf.keras.layers.Dense(256, activation='relu')(x)
            shared_features = tf.keras.layers.Dropout(0.1)(shared_features)

            # Update action output layer to ensure correct dimensions
            action_logits = tf.keras.layers.Dense(256, activation='relu')(shared_features)
            action_logits = tf.keras.layers.Dropout(0.1)(action_logits)
            # Action output layer - explicitly ensure 8 dimensions
            action_outputs = tf.keras.layers.Dense(8, activation='linear', name="action_outputs")(shared_features)

            outputs = [action_outputs]

            # Add parameter prediction heads if enabled
            if self.parameter_support:
                # Add parameter prediction heads for specific actions
                trajectory_params = tf.keras.layers.Dense(4, activation='tanh', name="trajectory_params")(
                    shared_features)
                velocity_params = tf.keras.layers.Dense(1, activation='sigmoid', name="velocity_params")(
                    shared_features)
                investigation_params = tf.keras.layers.Dense(2, activation='sigmoid', name="investigation_params")(
                    shared_features)
                refuel_params = tf.keras.layers.Dense(1, activation='sigmoid', name="refuel_params")(shared_features)

                outputs.extend([trajectory_params, velocity_params, investigation_params, refuel_params])

            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Different loss for different outputs
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss={
                    'action_outputs': tf.keras.losses.MeanSquaredError(),
                    'trajectory_params': tf.keras.losses.MeanSquaredError(),
                    'velocity_params': tf.keras.losses.MeanSquaredError(),
                    'investigation_params': tf.keras.losses.MeanSquaredError(),
                    'refuel_params': tf.keras.losses.MeanSquaredError()
                }
            )

            return model

    def predict(self, state_sequence):
        """Predict Q-values and parameters with support for variable sequence lengths."""
        # Convert to numpy array and ensure consistent format
        if isinstance(state_sequence, list):
            state_sequence = np.array(state_sequence)

        # Ensure at least 2D
        if len(state_sequence.shape) == 1:
            state_sequence = np.expand_dims(state_sequence, axis=-1)

        # Add batch dimension if not present
        has_batch_dim = len(state_sequence.shape) >= 3
        if not has_batch_dim:
            state_sequence = np.expand_dims(state_sequence, axis=0)

        batch_size = state_sequence.shape[0]

        # Now we handle each sequence in the batch appropriately
        padded_sequences = []
        for i in range(batch_size):
            seq = state_sequence[i]

            # First adjust to current sequence length
            if len(seq) > self.current_sequence_length:
                seq = seq[-self.current_sequence_length:]
            elif len(seq) < self.current_sequence_length:
                padding_needed = self.current_sequence_length - len(seq)
                padding = np.zeros((padding_needed, seq.shape[1]), dtype=np.float32)
                seq = np.vstack((padding, seq))

            # Then ensure it matches model's expected length
            if len(seq) > self.max_sequence_length:
                seq = seq[-self.max_sequence_length:]
            elif len(seq) < self.max_sequence_length:
                padding_needed = self.max_sequence_length - len(seq)
                padding = np.zeros((padding_needed, seq.shape[1]), dtype=np.float32)
                seq = np.vstack((padding, seq))

            padded_sequences.append(seq)

        # Stack all sequences into a batch
        state_sequence = np.array(padded_sequences)

        # Get raw predictions
        predictions = self.model.predict(state_sequence, verbose=0)

        # Format predictions as expected
        if self.parameter_support and isinstance(predictions, list):
            action_values = predictions[0][0]  # First element, remove batch dimension

            # Handle single value case
            if len(action_values.shape) == 0 or action_values.size == 1:
                action_values = np.full(self.num_actions, float(
                    action_values.item() if hasattr(action_values, 'item') else action_values))
                logging.warning(f"Model output dimension mismatch: got 1, expected {self.num_actions}")

            # Process parameter predictions if available
            parameter_values = {}
            if len(predictions) > 1:
                # Process parameters (existing code)
                # ...
                return action_values, parameter_values

            return action_values
        else:
            # For non-parameter case just return action values
            action_values = predictions[0] if isinstance(predictions, list) else predictions

            # Remove batch dimension if present
            if len(action_values.shape) > 1:
                action_values = action_values[0]

            # Ensure proper shape
            if len(action_values.shape) == 0 or action_values.size == 1:
                action_values = np.full(self.num_actions, float(
                    action_values.item() if hasattr(action_values, 'item') else action_values))

            return action_values


    def get_parameters_for_action(self, action_index, parameter_values):
        """Get appropriate parameters for a given action."""
        action_names = [
            'maintain_course', 'adjust_trajectory', 'increase_velocity',
            'decrease_velocity', 'investigate_anomaly', 'refuel', 'emergency_protocol'
        ]

        if action_index < 0 or action_index >= len(action_names):
            return {}

        action_name = action_names[action_index]

        if action_name in parameter_values:
            return parameter_values[action_name]

        return {}

    def train(self, state_sequences, targets, parameter_targets=None):
        """Train with support for variable sequence lengths and parameter prediction."""
        # Convert inputs to numpy arrays
        state_sequences = np.array(state_sequences)
        targets = np.array(targets)

        # Handle 3D targets by taking first slice or average
        if len(targets.shape) == 3 and targets.shape[2] > 0:  # Shape [batch,8,15]
            targets = targets[:, :, 0]  # Take first slice to get [batch,8]

        # Ensure proper 2D shape [batch_size, 8]
        if len(targets.shape) == 1:
            targets = np.expand_dims(targets, axis=-1)
        if targets.shape[-1] == 1:
            targets = np.broadcast_to(targets, (targets.shape[0], 8))

        # Process sequences to match expected dimensions
        adjusted_sequences = []
        for seq in state_sequences:
            seq = np.array(seq)
            if len(seq.shape) == 1:
                seq = np.expand_dims(seq, axis=-1)

            # Pad or truncate to match max sequence length
            if len(seq) > self.max_sequence_length:
                seq = seq[-self.max_sequence_length:]
            elif len(seq) < self.max_sequence_length:
                padding = np.zeros((self.max_sequence_length - len(seq), seq.shape[1]))
                seq = np.vstack((padding, seq))

            adjusted_sequences.append(seq)

        adjusted_sequences = np.array(adjusted_sequences)

        # Create parameter targets if needed
        batch_size = len(adjusted_sequences)
        if self.parameter_support:
            if parameter_targets is None:
                parameter_targets = {}

            # Prepare default parameter values with proper shapes
            param_dict = {
                'action_outputs': targets,
                'trajectory_params': parameter_targets.get('trajectory_params', np.zeros((batch_size, 4))),
                'velocity_params': parameter_targets.get('velocity_params', np.zeros((batch_size, 1))),
                'investigation_params': parameter_targets.get('investigation_params', np.zeros((batch_size, 2))),
                'refuel_params': parameter_targets.get('refuel_params', np.zeros((batch_size, 1)))
            }

            # Train with all outputs including parameters
            return self.model.fit(
                adjusted_sequences,
                param_dict,
                batch_size=min(64, batch_size),
                epochs=1,
                verbose=0
            )
        else:
            # Train with only action outputs
            return self.model.fit(
                adjusted_sequences,
                targets,
                batch_size=min(64, batch_size),
                epochs=1,
                verbose=0
            )

    def save_model(self, filepath="ai_captain_model"):
        """Save the model to disk."""
        self.model.save(f"{filepath}.keras")
        logging.info(f"Model saved to {filepath}.keras")

    def load_model(self, filepath="ai_captain_model"):
        """Load the model from disk."""
        try:
            if os.path.exists(f"{filepath}.keras"):
                self.model = tf.keras.models.load_model(f"{filepath}.keras")
                logging.info(f"Model loaded from {filepath}.keras")
            else:
                self.model = tf.keras.models.load_model(f"{filepath}.h5")
                logging.info(f"Model loaded from {filepath}.h5")
            return True
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return False



class MissionPhase:
    """Defines different mission phases with specific requirements."""

    PHASES = {
        "LAUNCH": {
            "sequence_length": 5,  # Short-term focus during critical launch
            "description": "Initial launch phase requiring rapid decisions",
            "requirements": {
                "fuel_minimum": 0.7,
                "anomaly_tolerance": 0.0  # Zero tolerance for anomalies
            }
        },
        "TRANSIT": {
            "sequence_length": 15,  # Medium-term focus during normal operations
            "description": "Steady transit between destinations",
            "requirements": {
                "fuel_minimum": 0.3,
                "anomaly_tolerance": 0.1  # Some tolerance for minor anomalies
            }
        },
        "EXPLORATION": {
            "sequence_length": 25,  # Longer historical context for exploration
            "description": "Active exploration of target areas",
            "requirements": {
                "fuel_minimum": 0.4,
                "anomaly_tolerance": 0.2  # Higher tolerance for anomalies during exploration
            }
        },
        "CRITICAL": {
            "sequence_length": 8,  # Medium-short for balanced decision-making
            "description": "Critical operation requiring careful but swift decisions",
            "requirements": {
                "fuel_minimum": 0.5,
                "anomaly_tolerance": 0.05  # Low tolerance for anomalies
            }
        },
        "EMERGENCY": {
            "sequence_length": 3,  # Very short for immediate response
            "description": "Emergency situation requiring immediate action",
            "requirements": {
                "fuel_minimum": 0.1,  # Lower requirements due to emergency
                "anomaly_tolerance": 0.3  # Higher tolerance as survival is a priority
            }
        }
    }

    @staticmethod
    def determine_phase(sensor_data, current_phase=None):
        """
        Determine the appropriate mission phase based on sensor data.
        Returns the phase name and properties.
        """
        # Default to TRANSIT if no current phase
        if current_phase is None:
            current_phase = "TRANSIT"

        # Extract relevant metrics
        fuel_level = sensor_data.get('fuel_level', 1.0)
        anomaly = sensor_data.get('anomaly_detector', False)
        velocity = sensor_data.get('velocity', np.array([0, 0, 0]))
        speed = np.linalg.norm(velocity)

        # Check for emergency conditions
        if fuel_level < 0.15 or (anomaly and fuel_level < 0.3):
            return "EMERGENCY", MissionPhase.PHASES["EMERGENCY"]

        # Check for critical conditions
        if fuel_level < 0.25 or (anomaly and current_phase != "EXPLORATION"):
            return "CRITICAL", MissionPhase.PHASES["CRITICAL"]

        # Check for launch conditions (high acceleration and early mission time)
        mission_time = sensor_data.get('mission_time', 0)
        if mission_time < 2.0 and speed > 1.0:  # First 2 hours and high speed
            return "LAUNCH", MissionPhase.PHASES["LAUNCH"]

        # Determine if in exploration mode based on speed and position
        position = sensor_data.get('position', np.array([0, 0, 0]))
        position_magnitude = np.linalg.norm(position)

        if speed < 0.5 and position_magnitude > 10.0:  # Slow movement far from origin
            return "EXPLORATION", MissionPhase.PHASES["EXPLORATION"]

        # Default to transit
        return "TRANSIT", MissionPhase.PHASES["TRANSIT"]


class DynamicTransformerModel(TransformerModel):
    """
    Enhanced Transformer model with dynamic sequence length based on mission phase
    and support for parameterised commands.
    """

    def __init__(self, input_dim, num_actions, max_sequence_length=25, parameter_support=True):
        self.max_sequence_length = max_sequence_length
        self.current_sequence_length = 10  # Default
        self.current_phase = "TRANSIT"  # Default phase
        # Initialise the parent class with max_sequence_length to ensure the model can handle max size
        super().__init__(input_dim, num_actions, sequence_length=max_sequence_length,
                         parameter_support=parameter_support)

    def predict(self, state_sequence):
        """
        Predict Q-values and parameters with support for variable sequence lengths.
        Adapts the input sequence to match what the model expects.
        """
        # First, adjust sequence to current mission phase length
        if len(state_sequence) > self.current_sequence_length:
            # Use the most recent data if we have more than needed for the current phase
            state_sequence = state_sequence[-self.current_sequence_length:]

        # Pad sequence if too short for the current phase
        elif len(state_sequence) < self.current_sequence_length:
            padding_needed = self.current_sequence_length - len(state_sequence)
            padding = np.zeros((padding_needed, state_sequence.shape[1]), dtype=np.float32)
            state_sequence = np.vstack((padding, state_sequence))

        # Now, ensure the sequence matches what the model expects (max_sequence_length)
        if len(state_sequence) > self.max_sequence_length:
            # Use the most recent data
            state_sequence = state_sequence[-self.max_sequence_length:]
        elif len(state_sequence) < self.max_sequence_length:
            # Additional padding needed
            padding_needed = self.max_sequence_length - len(state_sequence)
            padding = np.zeros((padding_needed, state_sequence.shape[1]), dtype=np.float32)
            state_sequence = np.vstack((padding, state_sequence))

        # Add a batch dimension if needed
        if len(state_sequence.shape) == 2:
            state_sequence = np.expand_dims(state_sequence, axis=0)

        # Create a dataset from the input array
        dataset = tf.data.Dataset.from_tensor_slices(state_sequence)

        # Set the auto-sharding policy
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)

        # Batch the dataset
        dataset = dataset.batch(1)

        # Use model.predict with the dataset
        return super().predict(state_sequence)

    def train(self, state_sequences, targets, parameter_targets=None):
        """Train with support for variable sequence lengths and parameter prediction."""
        # Ensure state_sequences is numpy array
        state_sequences = np.array(state_sequences)

        # Ensure targets has correct shape [batch_size, 8]
        targets = np.array(targets)
        if len(targets.shape) == 1:
            targets = np.expand_dims(targets, axis=0)
        if targets.shape[-1] == 1:
            targets = np.broadcast_to(targets, (targets.shape[0], 8))

        # Process and pad sequences to match expected dimensions
        adjusted_sequences = []
        batch_size = len(state_sequences)

        for seq in state_sequences:
            # Convert to numpy array if needed
            seq = np.array(seq)

            # Ensure 2D shape
            if len(seq.shape) == 1:
                seq = np.expand_dims(seq, axis=-1)

            # Pad or truncate sequence
            if len(seq) > self.max_sequence_length:
                seq = seq[-self.max_sequence_length:]
            elif len(seq) < self.max_sequence_length:
                padding = np.zeros((self.max_sequence_length - len(seq), seq.shape[1]))
                seq = np.vstack((padding, seq))

            adjusted_sequences.append(seq)

        # Convert to numpy array with correct shape [batch_size, sequence_length, input_dim]
        adjusted_sequences = np.array(adjusted_sequences)

        # Ensure targets match batch size
        targets = targets[:len(adjusted_sequences)]

        # Call parent's train method with adjusted data
        return super().train(adjusted_sequences, targets, parameter_targets)


class MissionMemorySystem:
    """Advanced memory system for missions with episodic and semantic memories."""

    def __init__(self, max_episodes=100, similarity_threshold=0.7):
        # Episodic memory: stores specific experiences/events
        self.episodic_memory = []
        self.max_episodes = max_episodes

        # Semantic memory: stores abstract knowledge and concepts
        self.semantic_memory = {
            "hazards": {},  # Knowledge about hazards
            "efficiency": {},  # Knowledge about efficient actions
            "celestial_bodies": {},  # Knowledge about planets, moons, etc.
            "mission_patterns": {}  # Learned patterns and behaviours
        }

        # Indices for efficient retrieval
        self.temporal_index = {}  # Time-based index
        self.spatial_index = {}  # Location-based index
        self.contextual_index = {}  # Context-based index

        self.similarity_threshold = similarity_threshold
        self.memory_utilization = 0.0  # Track how effectively memory is being used

    def store_episode(self, state, action, reward, phase, position, anomaly_detected=False, timestamp=None):
        """Store a specific mission episode/experience."""
        if timestamp is None:
            timestamp = time.time()

        episode = {
            "timestamp": timestamp,
            "state": state.copy() if hasattr(state, "copy") else state,
            "action": action,
            "reward": reward,
            "phase": phase,
            "position": position.copy() if hasattr(position, "copy") else position,
            "anomaly_detected": anomaly_detected,
            "retrieval_count": 0,  # Track how often this memory is retrieved
            "importance": abs(reward)  # Initial importance based on reward magnitude
        }

        # Add to episodic memory
        self.episodic_memory.append(episode)

        # Update indices
        position_key = tuple(np.round(position / 1e10))  # Spatial index key
        if position_key not in self.spatial_index:
            self.spatial_index[position_key] = []
        self.spatial_index[position_key].append(len(self.episodic_memory) - 1)

        # Update temporal index (by mission phase)
        if phase not in self.temporal_index:
            self.temporal_index[phase] = []
        self.temporal_index[phase].append(len(self.episodic_memory) - 1)

        # Update contextual index (by action and anomaly status)
        context_key = (action, anomaly_detected)
        if context_key not in self.contextual_index:
            self.contextual_index[context_key] = []
        self.contextual_index[context_key].append(len(self.episodic_memory) - 1)

        # Limit memory size by removing least important episodes
        if len(self.episodic_memory) > self.max_episodes:
            self._consolidate_memory()

        # Update semantic memory based on this episode
        self._update_semantic_memory(episode)

        return True

    def _consolidate_memory(self):
        """Remove least important memories when capacity is reached and properly update all indices."""
        # Sort episodes by importance
        importance_scores = [(i, ep["importance"]) for i, ep in enumerate(self.episodic_memory)]
        importance_scores.sort(key=lambda x: x[1])

        # Get the least important episode's index
        idx_to_remove = importance_scores[0][0]
        removed_episode = self.episodic_memory.pop(idx_to_remove)

        # Remove from spatial index
        position_key = tuple(np.round(removed_episode["position"] / 1e10))
        if position_key in self.spatial_index:
            if idx_to_remove in self.spatial_index[position_key]:
                self.spatial_index[position_key].remove(idx_to_remove)
            # Remove empty lists
            if not self.spatial_index[position_key]:
                del self.spatial_index[position_key]

        # Remove from temporal index
        phase = removed_episode["phase"]
        if phase in self.temporal_index:
            if idx_to_remove in self.temporal_index[phase]:
                self.temporal_index[phase].remove(idx_to_remove)
            # Remove empty lists
            if not self.temporal_index[phase]:
                del self.temporal_index[phase]

        # Remove from contextual index
        context_key = (removed_episode["action"], removed_episode["anomaly_detected"])
        if context_key in self.contextual_index:
            if idx_to_remove in self.contextual_index[context_key]:
                self.contextual_index[context_key].remove(idx_to_remove)
            # Remove empty lists
            if not self.contextual_index[context_key]:
                del self.contextual_index[context_key]

        # Update all indices to account for shifted indices
        # (after removing an item, all subsequent items shift down by 1)
        for key in self.spatial_index:
            self.spatial_index[key] = [i if i < idx_to_remove else i - 1 for i in self.spatial_index[key]]

        for key in self.temporal_index:
            self.temporal_index[key] = [i if i < idx_to_remove else i - 1 for i in self.temporal_index[key]]

        for key in self.contextual_index:
            self.contextual_index[key] = [i if i < idx_to_remove else i - 1 for i in self.contextual_index[key]]

        # Extract important knowledge into semantic memory before discarding
        if removed_episode["importance"] > 0.5 and removed_episode["retrieval_count"] > 0:
            # This was somewhat important, so preserve its knowledge
            self._extract_semantic_knowledge(removed_episode)

        return True

    def _extract_semantic_knowledge(self, episode):
        """Extract and preserve particularly important knowledge from an episode being removed."""
        # Extract any highly rewarded actions in specific phases
        if abs(episode["reward"]) > 1.0:
            action = episode["action"]
            phase = episode["phase"]

            # Add to mission patterns with an emphasis marker
            pattern_key = f"important_{phase}_{action}"
            if pattern_key not in self.semantic_memory["mission_patterns"]:
                self.semantic_memory["mission_patterns"][pattern_key] = {"count": 0, "reward": 0}

            self.semantic_memory["mission_patterns"][pattern_key]["count"] += 1
            self.semantic_memory["mission_patterns"][pattern_key]["reward"] += episode["reward"]

    def _rebuild_indices(self):
        """Rebuild all memory indices after consolidation."""
        # Reset indices
        self.spatial_index = {}
        self.temporal_index = {}
        self.contextual_index = {}

        # Rebuild indices
        for i, episode in enumerate(self.episodic_memory):
            # Spatial index
            position_key = tuple(np.round(episode["position"] / 1e10))
            if position_key not in self.spatial_index:
                self.spatial_index[position_key] = []
            self.spatial_index[position_key].append(i)

            # Temporal index
            phase = episode["phase"]
            if phase not in self.temporal_index:
                self.temporal_index[phase] = []
            self.temporal_index[phase].append(i)

            # Contextual index
            context_key = (episode["action"], episode["anomaly_detected"])
            if context_key not in self.contextual_index:
                self.contextual_index[context_key] = []
            self.contextual_index[context_key].append(i)

    def _update_semantic_memory(self, episode):
        """Extract abstract knowledge from episodes and update semantic memory."""
        # Update hazard knowledge
        if episode["anomaly_detected"]:
            action = episode["action"]
            reward = episode["reward"]
            phase = episode["phase"]

            hazard_key = f"response_{action}_{phase}"
            if hazard_key not in self.semantic_memory["hazards"]:
                self.semantic_memory["hazards"][hazard_key] = {"count": 0, "total_reward": 0}

            self.semantic_memory["hazards"][hazard_key]["count"] += 1
            self.semantic_memory["hazards"][hazard_key]["total_reward"] += reward

        # Update efficiency knowledge
        action = episode["action"]
        reward = episode["reward"]
        fuel_level = episode["state"][6] if len(episode["state"]) > 6 else 0

        efficiency_key = f"{action}_fuel_{int(fuel_level * 10)}"
        if efficiency_key not in self.semantic_memory["efficiency"]:
            self.semantic_memory["efficiency"][efficiency_key] = {"count": 0, "total_reward": 0}

        self.semantic_memory["efficiency"][efficiency_key]["count"] += 1
        self.semantic_memory["efficiency"][efficiency_key]["total_reward"] += reward

        # Update mission patterns based on phase transitions
        if len(self.episodic_memory) > 1:
            prev_episode = self.episodic_memory[-2]
            if prev_episode["phase"] != episode["phase"]:
                transition = f"{prev_episode['phase']}_to_{episode['phase']}"
                if transition not in self.semantic_memory["mission_patterns"]:
                    self.semantic_memory["mission_patterns"][transition] = 0
                self.semantic_memory["mission_patterns"][transition] += 1

    def retrieve_by_similarity(self, current_state, top_k=3):
        """Retrieve episodes most similar to current state."""
        if not self.episodic_memory:
            return []

        similarities = []
        for i, episode in enumerate(self.episodic_memory):
            # Calculate similarity between current state and episode state
            # Using cosine similarity for state vectors
            if isinstance(episode["state"], np.ndarray) and isinstance(current_state, np.ndarray):
                similarity = np.dot(episode["state"], current_state) / (
                        np.linalg.norm(episode["state"]) * np.linalg.norm(current_state))
            else:
                # Fallback similarity for non-array states
                similarity = 0.5  # Default medium similarity

            similarities.append((i, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k similar episodes
        results = []
        for i, sim in similarities[:top_k]:
            if sim >= self.similarity_threshold:
                episode = self.episodic_memory[i]
                # Update retrieval statistics
                episode["retrieval_count"] += 1
                episode["importance"] += 0.1  # Increase importance when retrieved
                results.append((episode, sim))

        return results

    def retrieve_by_context(self, action=None, anomaly=None, phase=None):
        """Retrieve episodes matching the given context."""
        matches = set()

        # Retrieve by action + anomaly context
        if action is not None and anomaly is not None:
            context_key = (action, anomaly)
            if context_key in self.contextual_index:
                matches.update(self.contextual_index[context_key])

        # Retrieve by phase
        if phase is not None and phase in self.temporal_index:
            if len(matches) > 0:
                # Intersection with existing matches
                matches = matches.intersection(set(self.temporal_index[phase]))
            else:
                matches = set(self.temporal_index[phase])

        # Convert indices to episodes
        results = []
        for idx in matches:
            if idx < len(self.episodic_memory):
                episode = self.episodic_memory[idx]
                episode["retrieval_count"] += 1
                episode["importance"] += 0.1
                results.append(episode)

        return results

    def retrieve_by_location(self, position, radius=1e10):
        """Retrieve episodes that happened near the given position."""
        position_key = tuple(np.round(position / 1e10))
        results = []

        # Check exact position first
        if position_key in self.spatial_index:
            for idx in self.spatial_index[position_key]:
                episode = self.episodic_memory[idx]
                episode["retrieval_count"] += 1
                episode["importance"] += 0.1
                results.append(episode)

        # If no exact matches or radius is larger, check nearby
        if not results or radius > 1e10:
            for episode in self.episodic_memory:
                dist = np.linalg.norm(episode["position"] - position)
                if dist <= radius:
                    if episode not in results:  # Avoid duplicates
                        episode["retrieval_count"] += 1
                        episode["importance"] += 0.1
                        results.append(episode)

        return results

    def get_semantic_knowledge(self, category, key=None):
        """Retrieve semantic knowledge from memory."""
        if category not in self.semantic_memory:
            return None

        if key is not None:
            return self.semantic_memory[category].get(key)

        return self.semantic_memory[category]

    def calculate_memory_utilization(self):
        """Calculate how effectively the memory system is being used."""
        if not self.episodic_memory:
            return 0.0

        # Calculate based on retrieval distribution and semantic memory enrichment
        retrieval_counts = [ep["retrieval_count"] for ep in self.episodic_memory]
        avg_retrieval = sum(retrieval_counts) / len(retrieval_counts) if retrieval_counts else 0

        # Count semantic memory entries
        semantic_entries = sum(len(entries) for entries in self.semantic_memory.values())

        # Combine metrics
        retrieval_factor = min(1.0, avg_retrieval / 5)  # Normalize, cap at 1.0
        semantic_factor = min(1.0, semantic_entries / 100)  # Normalize, cap at 1.0

        utilization = 0.6 * retrieval_factor + 0.4 * semantic_factor
        self.memory_utilization = utilization

        return utilization