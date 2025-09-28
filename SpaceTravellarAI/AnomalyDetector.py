import tensorflow as tf
import numpy as np
import logging
import json

class AnomalyDetector:
    """ Use An autoencoder to detect anomalies in the sensor data """

    def __init__(self, input_dim, latent_dim=4, threshold_multiplier=3.0):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.threshold_multiplier = threshold_multiplier
        self.model = self._build_model()
        self.reconstruction_errors = []
        self.anomaly_threshold = None

    def _build_model(self):
        """Build the autoencoder model."""
        # Encoder
        inputs = tf.keras.layers.Input(shape=(self.input_dim,))
        encoded = tf.keras.layers.Dense(16, activation='relu')(inputs)
        encoded = tf.keras.layers.Dense(8, activation='relu')(encoded)
        encoded = tf.keras.layers.Dense(self.latent_dim, activation='relu')(encoded)

        # Decoder
        decoded = tf.keras.layers.Dense(8, activation='relu')(encoded)
        decoded = tf.keras.layers.Dense(16, activation='relu')(decoded)
        decoded = tf.keras.layers.Dense(self.input_dim, activation='sigmoid')(decoded)

        # Autoencoder
        autoencoder = tf.keras.Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')

        return autoencoder

    def optimize_threshold(self, mean, std, z_score=2.8, false_positive_weight=0.6):
        """
        Calculate optimized anomaly threshold with customisable sensitivity

        Args:
            mean: Mean error from normal data
            std: Standard deviation of error
            z_score: Number of standard deviations for a threshold
            false_positive_weight: Weight to balance false positives/negatives
                                  (higher = more tolerant, fewer false positives)

        Returns:
            Optimized anomaly threshold
        """
        base_threshold = mean + z_score * std

        # Apply weighting factor based on false positive tolerance
        adjusted_threshold = base_threshold * (1 + false_positive_weight * 0.1)

        return adjusted_threshold

    def get_mission_aware_threshold(self, mission_step, base_threshold):
        """
        Adjust threshold based on the mission phase

        Args:
            mission_step: Current step in the mission
            base_threshold: Base anomaly threshold

        Returns:
            Adjusted threshold based on mission phase
        """
        # Early mission phase - be more tolerant
        if mission_step < 1000:
            return base_threshold * 1.2
        # Critical mission phase - be more sensitive
        elif 2000 <= mission_step < 3000:
            return base_threshold * 0.9
        # Normal sensitivity for other phases
        else:
            return base_threshold

    def transfer_learning_update(self, new_normal_data, epochs=20):
        """Update the anomaly detector with new normal data patterns."""
        # keep a copy of the old thresholds
        original_threshold = self.anomaly_threshold

        # fine-tune on the new data with a lower learning rate
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
        self.model.fit(
            new_normal_data, new_normal_data,
            epochs=epochs,
            batch_size=32,
            verbose=0
        )

        # Compute new reconstruction errors
        reconstructed = self.model.predict(new_normal_data)
        errors = np.mean(np.square(new_normal_data - reconstructed), axis=1)

        # Update threshold with the weighted combination
        new_threshold = np.mean(errors) + self.threshold_multiplier * np.std(errors)
        self.anomaly_threshold = 0.7 * self.anomaly_threshold + 0.3 * new_threshold

        logging.info(
            f"Transfer learning complete. Threshold updated from {original_threshold} to {self.anomaly_threshold}")
        return True

    def train(self, normal_data, epochs=50, batch_size=32, validation_split=0.1, mission_step=None):
        """Train the autoencoder on normal data to learn the normal pattern."""
        history = self.model.fit(
            normal_data, normal_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )

        # Compute reconstruction errors on training data
        reconstructed = self.model.predict(normal_data)
        errors = np.mean(np.square(normal_data - reconstructed), axis=1)
        self.reconstruction_errors = errors

        # Calculate mean and std of errors
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        # Use optimised threshold calculation
        self.anomaly_threshold = self.optimize_threshold(mean_error, std_error)

        # Apply mission-aware adjustment if mission step is provided
        if mission_step is not None:
            self.anomaly_threshold = self.get_mission_aware_threshold(mission_step, self.anomaly_threshold)

        logging.info(f"Anomaly detector trained. Threshold: {self.anomaly_threshold}")
        return history

    def calculate_adaptive_threshold(self, recent_data, window_size=100):
        """
        Calculate adaptive threshold based on recent observations.

        Args:
            recent_data: Recent observations to compute threshold from
            window_size: Size of the sliding window

        Returns:
            Adaptive threshold based on recent data patterns
        """
        if len(recent_data) < window_size:
            return self.anomaly_threshold

        # Get most recent data
        recent_window = recent_data[-window_size:]

        # Compute reconstruction errors
        reconstructed = self.model.predict(recent_window, verbose=0)
        errors = np.mean(np.square(recent_window - reconstructed), axis=1)

        # Calculate mean and std
        recent_mean = np.mean(errors)
        recent_std = np.std(errors)

        # Calculate adaptive threshold
        base_adaptive = recent_mean + self.threshold_multiplier * recent_std

        # Blend with existing threshold
        return 0.7 * self.anomaly_threshold + 0.3 * base_adaptive

    def detect_anomaly(self, data, mission_step=None):
        """
        Detect if the given data contains an anomaly.
        Returns: (is_anomaly, anomaly_score)
        """
        # Ensure data is in batch format
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)

        # Calculate reconstruction error
        reconstructed = self.model.predict(data, verbose=0)
        error = np.mean(np.square(data - reconstructed), axis=1)

        # Apply the mission-aware threshold if mission step is provided
        threshold = self.anomaly_threshold
        if mission_step is not None:
            threshold = self.get_mission_aware_threshold(mission_step, threshold)

        # Determine if anomaly based on threshold
        is_anomaly = error > threshold

        return is_anomaly[0], error[0]  # Return scalar values

    def save_model(self, filepath="anomaly_detector_model"):
        """Save the model and threshold."""
        self.model.save(f"{filepath}.keras")
        with open(f"{filepath}_threshold.json", "w") as f:
            json.dump({
                "anomaly_threshold": float(self.anomaly_threshold),
                "mean_error": float(np.mean(self.reconstruction_errors)),
                "std_error": float(np.std(self.reconstruction_errors))
            }, f)

    def load_model(self, filepath="anomaly_detector_model"):
        """Load the model and threshold."""
        try:
            self.model = tf.keras.models.load_model(f"{filepath}.keras")
            with open(f"{filepath}_threshold.json", "r") as f:
                threshold_data = json.loads(f.read())
                self.anomaly_threshold = threshold_data["anomaly_threshold"]
            return True
        except Exception as e:
            logging.error(f"Failed to load anomaly detector model: {e}")
            return False



class EnvironmentalHazardDetector(AnomalyDetector):
    """Specialised anomaly detector for environmental hazards based on the autoencoder architecture."""

    def __init__(self, input_dim, latent_dim=4, threshold_multiplier=2.5):
        super().__init__(input_dim, latent_dim, threshold_multiplier)
        # Hazard categories with their specific thresholds
        self.hazard_thresholds = {
            "radiation": 0.0,  # Will be set after training
            "temperature": 0.0,
            "pressure": 0.0,
            "toxicity": 0.0
        }

    def train_on_environment_data(self, normal_data, hazard_labels=None, mission_step=None):
        """Train the detector with environment-specific data."""
        history = self.train(normal_data, mission_step=mission_step)

        # If we have labelled data for different environmental aspects
        if hazard_labels and isinstance(hazard_labels, dict):
            for hazard_type, data in hazard_labels.items():
                if hazard_type in self.hazard_thresholds:
                    reconstructed = self.model.predict(data)
                    errors = np.mean(np.square(data - reconstructed), axis=1)

                    # Calculate mean and std for this hazard type
                    mean_error = np.mean(errors)
                    std_error = np.std(errors)

                    # Use the optimized threshold calculation
                    optimized_threshold = self.optimize_threshold(
                        mean_error,
                        std_error,
                        z_score=2.5  # Using the threshold_multiplier from the constructor
                    )

                    # Apply mission-aware adjustment if mission step is provided
                    if mission_step is not None:
                        optimized_threshold = self.get_mission_aware_threshold(
                            mission_step,
                            optimized_threshold
                        )

                    self.hazard_thresholds[hazard_type] = optimized_threshold

        return history

    def detect_hazard(self, data, hazard_type=None, mission_step=None):
        """
        Detect if environmental data indicates a hazard.

        Args:
            data: The environmental sensor data
            hazard_type: Optional specific hazard type to check
            mission_step: Current step in the mission for context-aware thresholds

        Returns:
            (is_hazard, hazard_score, hazard_type) if hazard_type is None
            (is_hazard, hazard_score) if a specific hazard_type is provided
        """
        # Ensure data is in batch format
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)

        # Calculate reconstruction error
        reconstructed = self.model.predict(data, verbose=0)
        error = np.mean(np.square(data - reconstructed), axis=1)

        # If checking for the specific hazard type
        if hazard_type and hazard_type in self.hazard_thresholds:
            threshold = self.hazard_thresholds[hazard_type]

            # Apply mission-aware adjustment if mission step is provided
            if mission_step is not None and threshold is not None:
                threshold = self.get_mission_aware_threshold(mission_step, threshold)

            # Safety check - if threshold is None, use default value
            if threshold is None:
                threshold = 0.8  # Default fallback threshold
                logging.warning(f"Using default threshold 0.8 for hazard detection as no threshold was set")

            is_hazard = error > threshold
            return is_hazard[0], error[0]

        # Get the general anomaly threshold with mission context if provided
        threshold = self.anomaly_threshold

        # Safety check - if threshold is None, use default value
        if threshold is None:
            threshold = 0.8  # Default fallback threshold
            logging.warning(f"Using default threshold 0.8 for general hazard detection as no threshold was set")

        if mission_step is not None and threshold is not None:
            threshold = self.get_mission_aware_threshold(mission_step, threshold)

        # Check against the threshold
        is_hazard = error > threshold

        # Determine a hazard type if applicable
        detected_hazard = None
        if is_hazard[0]:
            # Find which environmental aspect contributed most to the error
            feature_errors = np.square(data - reconstructed)[0]
            hazard_indices = {
                "radiation": [2, 3],  # Example indices where radiation data is stored
                "temperature": [0, 1],
                "pressure": [4, 5],
                "toxicity": [6, 7]
            }

            max_error = 0
            for htype, indices in hazard_indices.items():
                avg_error = np.mean(feature_errors[indices])
                if avg_error > max_error:
                    max_error = avg_error
                    detected_hazard = htype

        return is_hazard[0], error[0], detected_hazard


class EnsembleAnomalyDetector:
    """Combines multiple anomaly detectors for more robust detection."""

    def __init__(self, detectors=None, voting_threshold=0.5):
        """
        Initialize the ensemble detector.

        Args:
            detectors: List of anomaly detector instances
            voting_threshold: Fraction of detectors needed to declare an anomaly
        """
        self.detectors = detectors if detectors else []
        self.voting_threshold = voting_threshold

    def add_detector(self, detector):
        """Add a detector to the ensemble."""
        self.detectors.append(detector)

    def detect_anomaly(self, data, mission_step=None):
        """
        Detect anomalies using all detectors in the ensemble.

        Args:
            data: Input data to check for anomalies
            mission_step: Current mission step for context

        Returns:
            (is_anomaly, confidence_score)
        """
        if not self.detectors:
            return False, 0.0

        # Collect votes and scores from all detectors
        votes = []
        scores = []

        for detector in self.detectors:
            is_anomaly, score = detector.detect_anomaly(data, mission_step)
            votes.append(is_anomaly)
            scores.append(score)

        # Calculate the fraction of positive votes
        vote_fraction = sum(votes) / len(votes)

        # Calculate weighted anomaly score
        weighted_score = sum(scores) / len(scores)

        # Determine if it's an anomaly based on voting threshold
        is_ensemble_anomaly = vote_fraction >= self.voting_threshold

        return is_ensemble_anomaly, weighted_score

    def save_ensemble(self, filepath="ensemble_detector"):
        """Save all detectors in the ensemble."""
        for i, detector in enumerate(self.detectors):
            detector.save_model(f"{filepath}_detector_{i}")

        # Save ensemble configuration
        with open(f"{filepath}_config.json", "w") as f:
            json.dump({
                "detector_count": len(self.detectors),
                "voting_threshold": self.voting_threshold
            }, f)

    def load_ensemble(self, filepath="ensemble_detector", detector_class=AnomalyDetector):
        """Load all detectors in the ensemble."""
        try:
            with open(f"{filepath}_config.json", "r") as f:
                config = json.load(f)
                self.voting_threshold = config["voting_threshold"]
                detector_count = config["detector_count"]

            self.detectors = []
            for i in range(detector_count):
                detector = detector_class(input_dim=8)  # Adjust input_dim as needed
                if detector.load_model(f"{filepath}_detector_{i}"):
                    self.detectors.append(detector)

            return len(self.detectors) > 0
        except Exception as e:
            logging.error(f"Failed to load ensemble detector: {e}")
            return False