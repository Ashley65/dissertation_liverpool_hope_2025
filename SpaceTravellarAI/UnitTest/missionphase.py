import unittest
from unittest.mock import MagicMock
from MissionPhaseAwareness import MissionPhaseAwareness
from astroDataManager import AstroDataManager

class TestMissionPhaseAwarenessIntegration(unittest.TestCase):
    def setUp(self):
        # Initialize MissionPhaseAwareness and mock dependencies
        self.phase_awareness = MissionPhaseAwareness()
        self.astro_data = MagicMock(spec=AstroDataManager)
        self.astro_data.celestial_objects = {
            "planet1": {"position": [100, 200, 300], "type": "planet"},
            "asteroid1": {"position": [400, 500, 600], "type": "asteroid"}
        }

    def test_phase_transition(self):
        # Simulate spacecraft status
        status_data = {
            "position": [150, 250, 350],
            "velocity": [0, 0, 0],
            "fuel_level": 0.5,
            "mission_time": 1000,
            "target_position": [400, 500, 600],
            "collision_warning": False,
            "system_failures": []
        }

        # Update phase awareness
        phase_data = self.phase_awareness.update(status_data)

        # Verify phase transition
        self.assertIn(phase_data["current_phase"], ["TRANSIT", "EXPLORATION", "EMERGENCY"])
        self.assertGreaterEqual(phase_data["phase_confidence"][phase_data["current_phase"]], 0.5)

    def test_integration_with_astro_data(self):
        # Simulate spacecraft status
        status_data = {
            "position": [150, 250, 350],
            "velocity": [0, 0, 0],
            "fuel_level": 0.5,
            "mission_time": 1000,
            "target_position": [400, 500, 600],
            "collision_warning": False,
            "system_failures": []
        }

        # Integrate with AstroDataManager
        self.phase_awareness.integrate_astro_data(self.astro_data, status_data)

        # Verify adjustments to phase confidence
        self.assertGreater(self.phase_awareness.phase_confidence["EXPLORATION"], 0.0)

if __name__ == "__main__":
    unittest.main()