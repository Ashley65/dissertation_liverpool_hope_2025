import unittest
from unittest.mock import patch, MagicMock
import time
import json
from collections import defaultdict
from networking import CommandHandler


class TestCommandHandler(unittest.TestCase):

    def setUp(self):
        """Set up a fresh CommandHandler for each test."""
        self.handler = CommandHandler()

    def test_add_command(self):
        """Test adding commands with different priorities."""
        # Add commands with different priorities
        self.handler.add_command("maintain_course")
        self.handler.add_command("emergency_protocol")
        self.handler.add_command("refuel")

        # Check queue length
        self.assertEqual(len(self.handler.command_queue), 3)

        # Verify priority ordering - emergency_protocol should be first
        self.assertEqual(self.handler.command_queue[0]["command"], "emergency_protocol")

    def test_add_unknown_command(self):
        """Test adding an unknown command without priority override."""
        result = self.handler.add_command("unknown_command")
        self.assertFalse(result)
        self.assertEqual(len(self.handler.command_queue), 0)

        # With priority override, it should work
        result = self.handler.add_command("unknown_command", priority_override=5)
        self.assertTrue(result)
        self.assertEqual(len(self.handler.command_queue), 1)

    def test_get_next_command(self):
        """Test getting commands in priority order with cooldown."""
        # Add commands
        self.handler.add_command("maintain_course")
        self.handler.add_command("refuel")

        # Get first command (should be refuel as it has higher priority)
        current_step = 10
        command = self.handler.get_next_command(current_step)
        self.assertEqual(command, "refuel")

        # Queue should have one command left
        self.assertEqual(len(self.handler.command_queue), 1)

        # Get next command
        command = self.handler.get_next_command(current_step)
        self.assertEqual(command, "maintain_course")

        # Queue should be empty now
        self.assertEqual(len(self.handler.command_queue), 0)

    def test_command_cooldown(self):
        """Test that commands respect cooldown periods."""
        # Add same command twice
        self.handler.add_command("refuel")
        self.handler.add_command("refuel")

        # Execute first command
        current_step = 10
        command = self.handler.get_next_command(current_step)
        self.assertEqual(command, "refuel")

        # Try to execute second command immediately (should return None due to cooldown)
        command = self.handler.get_next_command(current_step)
        self.assertIsNone(command)

        # Try again after cooldown period
        cooldown = self.handler.execution_cooldown["refuel"]
        command = self.handler.get_next_command(current_step + cooldown)
        self.assertEqual(command, "refuel")

    def test_record_outcome(self):
        """Test recording command outcomes."""
        # Record a successful outcome
        command = "adjust_trajectory"
        details = {"fuel_level": 0.75, "mission_phase": "TRANSIT"}

        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            self.handler.record_outcome(command, True, details)

        # Check that the outcome was recorded
        self.assertTrue(command in self.handler.command_outcomes)
        self.assertEqual(len(self.handler.command_outcomes[command]), 1)
        self.assertTrue(self.handler.command_outcomes[command][0]["success"])

        # Record a failure outcome for the same command
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            self.handler.record_outcome(command, False, {"reason": "insufficient fuel"})

        # Check both outcomes are recorded
        self.assertEqual(len(self.handler.command_outcomes[command]), 2)
        self.assertFalse(self.handler.command_outcomes[command][1]["success"])

    def test_get_command_success_rate(self):
        """Test calculating command success rates."""
        # Setup outcomes: 2 successes, 1 failure
        command = "investigate_anomaly"

        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            self.handler.record_outcome(command, True, {})
            self.handler.record_outcome(command, True, {})
            self.handler.record_outcome(command, False, {})

        # Calculate success rate
        success_rate = self.handler.get_command_success_rate(command)
        self.assertEqual(success_rate, 2 / 3)

        # Test for a command with no history
        self.assertIsNone(self.handler.get_command_success_rate("no_history"))

    def test_clear_queue(self):
        """Test clearing the command queue."""
        # Add some commands
        self.handler.add_command("maintain_course")
        self.handler.add_command("refuel")
        self.handler.add_command("emergency_protocol")

        # Clear queue
        queue_size = self.handler.clear_queue()

        # Verify queue is empty and correct size was returned
        self.assertEqual(queue_size, 3)
        self.assertEqual(len(self.handler.command_queue), 0)

    def test_add_mission(self):
        """Test adding a complex mission with actions."""
        mission_data = {"priority_level": "high", "target": "asteroid"}
        action_sequence = ["adjust_trajectory", "increase_velocity", "investigate_anomaly"]

        # Add mission
        self.handler.add_mission(mission_data, action_sequence)

        # Check it was added with correct priority
        self.assertEqual(len(self.handler.command_queue), 1)
        self.assertEqual(self.handler.command_queue[0]["priority"], 3)
        self.assertEqual(len(self.handler.command_queue[0]["actions"]), 3)

        # Test mission with low priority
        mission_data_low = {"priority_level": "low", "target": "moon"}
        self.handler.add_mission(mission_data_low, action_sequence)

        # High priority mission should still be first
        self.assertEqual(self.handler.command_queue[0]["priority"], 3)
        self.assertEqual(self.handler.command_queue[1]["priority"], 1)

    def test_execute_mission_actions_sequence(self):
        """Test executing a mission's action sequence in order."""
        mission_data = {"priority_level": "high", "target": "asteroid"}
        action_sequence = ["adjust_trajectory", "increase_velocity", "investigate_anomaly"]

        # Add mission
        self.handler.add_mission(mission_data, action_sequence)

        # Execute actions one by one
        current_step = 10
        first_action = self.handler.get_next_command(current_step)
        self.assertEqual(first_action, "adjust_trajectory")

        # Mission should still be in queue with two actions left
        self.assertEqual(len(self.handler.command_queue), 1)
        self.assertEqual(len(self.handler.command_queue[0]["actions"]), 2)

        # Get next action
        second_action = self.handler.get_next_command(current_step + 1)
        self.assertEqual(second_action, "increase_velocity")

        # Last action
        third_action = self.handler.get_next_command(current_step + 2)
        self.assertEqual(third_action, "investigate_anomaly")

        # Queue should be empty now
        self.assertEqual(len(self.handler.command_queue), 0)


if __name__ == '__main__':
    unittest.main()