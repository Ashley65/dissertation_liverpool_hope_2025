import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import json
import os


class TestAICaptain(unittest.TestCase):
    """Unit tests for the AICaptain class and its decision-making system."""

    def setUp(self):
        """Set up test environment before each test."""
        # Import here to avoid circular imports
        from main import AICaptain

        # Patch all dependencies
        self.sensor_data_patch = patch('main.EnhancedSensorData')
        self.network_comm_patch = patch('main.NetworkCommunication')
        self.command_handler_patch = patch('main.CommandHandler')
        self.phase_awareness_patch = patch('main.MissionPhaseAwareness')
        self.astro_data_patch = patch('main.AstroDataManager')
        self.fuel_optimizer_patch = patch('main.FuelOptimiser')
        self.ai_optimizer_patch = patch('main.AIOptimiser')

        # Start patches
        self.mock_sensor_data = self.sensor_data_patch.start()
        self.mock_network_comm = self.network_comm_patch.start()
        self.mock_command_handler = self.command_handler_patch.start()
        self.mock_phase_awareness = self.phase_awareness_patch.start()
        self.mock_astro_data = self.astro_data_patch.start()
        self.mock_fuel_optimizer = self.fuel_optimizer_patch.start()
        self.mock_ai_optimizer = self.ai_optimizer_patch.start()

        # Configure mocks
        self.mock_sensor_data.return_value.preprocess_for_model.return_value = np.zeros(10)
        self.mock_sensor_data.return_value.get_readings.return_value = {
            'fuel_level': 0.8,
            'temperature': 22.0,
            'velocity': np.array([1.0, 0.0, 0.0]),
            'position': np.array([0.0, 0.0, 0.0]),
            'mission_time': 1.0
        }
        self.mock_sensor_data.return_value.current_readings = {
            'fuel_level': 0.8,
            'temperature': 22.0,
            'velocity': np.array([1.0, 0.0, 0.0]),
            'position': np.array([0.0, 0.0, 0.0]),
            'mission_time': 1.0
        }

        # Mock phase awareness
        self.mock_phase_awareness.return_value.update.return_value = {
            "current_phase": "TRANSIT",
            "phase_confidence": {"TRANSIT": 0.9}
        }
        self.mock_phase_awareness.return_value.get_policy_recommendations.return_value = {
            "reward_modifiers": {"fuel_efficiency": 1.2, "speed": 0.8, "safety": 1.0},
            "action_preferences": {"maintain_course": 0.2}
        }

        # Initialize the captain with the mocked dependencies
        self.captain = AICaptain()

        # Mock the model's predict method
        self.captain.model.predict = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]))

        # Mock the anomaly detector
        self.captain.anomaly_detector.detect_anomaly = MagicMock(return_value=(False, 0.2))
        self.captain.anomaly_detector_trained = True

    def tearDown(self):
        """Clean up after each test."""
        self.sensor_data_patch.stop()
        self.network_comm_patch.stop()
        self.command_handler_patch.stop()
        self.phase_awareness_patch.stop()
        self.astro_data_patch.stop()
        self.fuel_optimizer_patch.stop()
        self.ai_optimizer_patch.stop()

    def test_initialization(self):
        """Test that the AICaptain initializes correctly."""
        self.assertEqual(self.captain.num_actions, 8)
        self.assertEqual(len(self.captain.actions), 8)
        self.assertEqual(self.captain.current_phase, "TRANSIT")
        self.assertEqual(self.captain.mission_step, 0)

    def test_get_action_exploitation(self):
        """Test the action selection in exploitation mode."""
        # Set epsilon to 0 for pure exploitation
        self.captain.epsilon = 0

        # Get an action
        action = self.captain.get_action(np.zeros(10))

        # Verify it chose the highest Q-value action (index 7)
        self.assertEqual(action, 7)

    def test_get_action_exploration(self):
        """Test the action selection in exploration mode."""
        # Set epsilon to 1 for pure exploration
        self.captain.epsilon = 1

        with patch('numpy.random.random', return_value=0.5), \
                patch('numpy.random.randint', return_value=3):
            # Get an action
            action = self.captain.get_action(np.zeros(10))

            # Verify it chose a random action (mocked to return 3)
            self.assertEqual(action, 3)

    def test_command_handler_priority(self):
        """Test that command handler takes priority when it has a command."""
        # Set up the command handler to return a command
        self.captain.command_handler.get_next_command = MagicMock(return_value="maintain_course")

        # Get an action
        action = self.captain.get_action(np.zeros(10))

        # Action should be the index of "maintain_course"
        self.assertEqual(action, 0)

    def test_anomaly_detection_response(self):
        """Test the AI's response to anomalies."""
        # Set anomaly detector to detect a critical anomaly
        self.captain.anomaly_detector.detect_anomaly = MagicMock(return_value=(True, 0.9))

        # Set an explicit anomaly threshold for testing
        self.captain.anomaly_detector.anomaly_threshold = 0.7

        # Mark the detector as trained
        self.captain.anomaly_detector_trained = True

        # Get an action
        action = self.captain.get_action(np.zeros(10))

        # Should choose emergency protocol (index 6)
        self.assertEqual(action, 4)

    def test_evaluate_action(self):
        """Test the reward calculation for actions."""
        state = np.zeros(10)

        # Test reward for maintaining course in normal conditions
        action_maintain = self.captain.action_ids["maintain_course"]
        reward_maintain = self.captain.evaluate_action(state, action_maintain)

        # Test reward for emergency action in normal conditions
        action_emergency = self.captain.action_ids["emergency_protocol"]
        reward_emergency = self.captain.evaluate_action(state, action_emergency)

        # Emergency protocol in normal conditions should have lower reward
        self.assertLess(reward_emergency, reward_maintain)

    def test_run_mission_step(self):
        """Test a complete mission step execution."""
        # Mock epsilon to ensure deterministic behavior
        self.captain.epsilon = 0

        # Run a mission step
        action_name, reward = self.captain.run_mission_step()

        # Verify mission step incremented
        self.assertEqual(self.captain.mission_step, 1)

        # Verify we got an action name back
        self.assertIn(action_name, self.captain.actions.values())

        # Verify reward is a number
        self.assertIsInstance(reward, (int, float))

    def test_save_load_checkpoint(self):
        """Test saving and loading model checkpoints."""
        # Set up test values
        self.captain.epsilon = 0.5
        self.captain.mission_step = 42

        # Create a temporary filename
        temp_file = "temp_checkpoint"

        # Mock save_model method
        self.captain.model.save_model = MagicMock(return_value=True)

        # Save checkpoint
        self.captain.save_checkpoint(temp_file)

        # Reset values
        self.captain.epsilon = 1.0
        self.captain.mission_step = 0

        # Mock load_model method
        self.captain.model.load_model = MagicMock(return_value=True)

        # Mock checkpoint file
        with patch("os.path.exists", return_value=True), \
                patch("builtins.open", return_value=MagicMock()):
            # Mock json.load to return our test values
            with patch("json.load", return_value={"epsilon": 0.5, "mission_step": 42}):
                result = self.captain.load_checkpoint(temp_file)
                self.assertTrue(result)
                self.assertEqual(self.captain.epsilon, 0.5)
                self.assertEqual(self.captain.mission_step, 42)

    def test_run_action(self):
        """Test running an action and getting success/failure result."""
        # Test refuel action
        refuel_action = self.captain.action_ids["refuel"]
        result = self.captain.run_action(refuel_action)

        # Should be a dictionary with success status
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("action", result)
        self.assertIn("details", result)

    def test_execute_command_with_tracking(self):
        """Test executing a command with outcome tracking."""
        # Mock run_action to return a success result
        self.captain.run_action = MagicMock(return_value={
            "success": True,
            "action": "maintain_course",
            "details": {"fuel_used": 0.01}
        })

        # Execute a command
        action = self.captain.action_ids["maintain_course"]
        result = self.captain.execute_command_with_tracking(action)

        # Verify the command_handler recorded the outcome
        self.captain.command_handler.record_outcome.assert_called_once()

        # Verify result matches expected format
        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "maintain_course")


if __name__ == "__main__":
    unittest.main()