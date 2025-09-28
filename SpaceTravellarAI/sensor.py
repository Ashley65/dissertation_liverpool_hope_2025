import numpy as np


class SensorData:
    """Handles the collection and preprocessing of spacecraft sensor data."""

    def __init__(self, sensor_config):
        self.sensor_config = sensor_config
        self.current_readings = {}

    def get_readings(self):
        """
        Get current readings from all sensors.
        In a real implementation, this would interface with actual spacecraft sensors.
        """
        # Simulate sensor readings for development
        self.current_readings = {
            'fuel_level': np.random.uniform(0.5, 1.0),
            'temperature': np.random.normal(22, 3),
            'radiation': np.random.exponential(0.5),
            'velocity': np.array([np.random.normal(0, 0.1) for _ in range(3)]),
            'position': np.array([np.random.normal(0, 0.1) for _ in range(3)]),
            'gravity': np.random.normal(9.8, 0.1),
            'anomaly_detector': np.random.random() > 0.95,  # Occasionally detect anomalies
        }
        return self.current_readings

    @staticmethod
    def preprocess_for_model(readings):
        """Transform raw sensor data into the format expected by the model."""
        # Convert readings to a fixed-length vector for the model
        features = []
        features.append(readings['fuel_level'])
        features.append(readings['temperature'])
        features.append(readings['radiation'])
        features.extend(readings['velocity'])
        features.extend(readings['position'])
        features.append(readings['gravity'])
        features.append(1.0 if readings['anomaly_detector'] else 0.0)

        return np.array(features, dtype=np.float32)

class EnhancedSensorData(SensorData):
    """Enhanced sensor simulation with more realistic temporal patterns."""

    def __init__(self, sensor_config):
        super().__init__(sensor_config)
        # Previous readings to create temporal coherence
        self.prev_readings = {
            'fuel_level': 1.0,
            'temperature': 22.0,
            'radiation': 0.5,
            'velocity': np.array([0.0, 0.0, 0.0]),
            'position': np.array([0.0, 0.0, 0.0]),
            'gravity': 9.8,
            'anomaly_detector': False
        }
        # Mission time tracking in simulated hours
        self.mission_time = 0
        # Fuel consumption rate per step
        self.fuel_consumption_rate = 0.001

    def get_readings(self):
        """Generate sensor readings with temporal coherence and physical relationships."""
        # Update mission time
        self.mission_time += 0.1  # 0.1-hour increment per step

        # Update fuel with consistent depletion and previous state
        fuel_level = max(0.0, self.prev_readings['fuel_level'] - self.fuel_consumption_rate)

        # Temperature varies based on previous + small random change
        temp_change = np.random.normal(0, 0.3)  # Small change each step
        temperature = self.prev_readings['temperature'] + temp_change

        # Radiation depends on position (e.g., higher in certain regions)
        position = self.prev_readings['position'] + self.prev_readings['velocity'] * 0.1
        position_magnitude = np.linalg.norm(position)
        radiation = 0.5 + 0.1 * np.sin(position_magnitude * 0.1) + np.random.exponential(0.1)

        # Velocity changes gradually
        velocity_change = np.array([np.random.normal(0, 0.01) for _ in range(3)])
        velocity = self.prev_readings['velocity'] + velocity_change

        # Gravity varies based on position
        gravity = 9.8 - 0.01 * position_magnitude + np.random.normal(0, 0.05)

        # Anomalies are rare but can persist for multiple steps
        if self.prev_readings['anomaly_detector']:
            # If already in anomaly, 80% chance to continue
            anomaly = np.random.random() > 0.2
        else:
            # 2% chance of new anomaly
            anomaly = np.random.random() > 0.98

        # Create the new readings
        self.current_readings = {
            'fuel_level': fuel_level,
            'temperature': temperature,
            'radiation': radiation,
            'velocity': velocity,
            'position': position,
            'gravity': gravity,
            'anomaly_detector': anomaly,
            'mission_time': self.mission_time
        }

        # Store as previous for next iteration
        self.prev_readings = self.current_readings.copy()

        return self.current_readings