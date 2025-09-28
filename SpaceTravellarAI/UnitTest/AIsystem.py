import unittest
from unittest.mock import patch, MagicMock
import json
import zmq
from test import CommandTestClient


class TestAISystem(unittest.TestCase):
    """Unit tests for the AI Captain system and CommandTestClient."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock ZMQ socket
        self.mock_socket = MagicMock()

        # Patch the ZMQ Context and socket creation
        self.context_patch = patch('zmq.Context')
        self.mock_context = self.context_patch.start()
        self.mock_context.return_value.socket.return_value = self.mock_socket

        # Initialise client with the mocked socket
        self.client = CommandTestClient("tcp://localhost:5555")

    def tearDown(self):
        """Clean up after each test."""
        self.context_patch.stop()

    def test_initialization(self):
        """Test client initialisation."""
        self.assertEqual(self.client.server_address, "tcp://localhost:5555")
        self.assertEqual(len(self.client.available_commands), 7)
        self.assertEqual(len(self.client.test_sequences), 5)

    def test_request_command_execution(self):
        """Test command execution request."""
        # Set up mock responses
        self.mock_socket.recv_string.side_effect = [
            json.dumps({"status": "ready"}),
            json.dumps({"status": "command_executed", "command": "maintain_course"})
        ]

        # Execute the command
        result = self.client.request_command_execution("maintain_course")

        # Check the result
        self.assertTrue(result)

        # Verify the correct messages were sent
        calls = self.mock_socket.send_string.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(json.loads(calls[0][0][0])["type"], "request_command")
        self.assertEqual(json.loads(calls[1][0][0])["command"], "maintain_course")

    def test_request_command_execution_failure(self):
        """Test command execution failure handling."""
        # Set up mock to raise an exception
        self.mock_socket.send_string.side_effect = zmq.ZMQError("Test error")

        # Execute the command
        result = self.client.request_command_execution("maintain_course")

        # Check the result
        self.assertFalse(result)

    def test_request_status(self):
        """Test status request."""
        # Set up mock response
        expected_response = {"status": "operational", "fuel": 90, "velocity": 5}
        self.mock_socket.recv_string.return_value = json.dumps(expected_response)

        # Request status
        result = self.client.request_status()

        # Check the result
        self.assertEqual(result, expected_response)

        # Verify the correct message was sent
        self.mock_socket.send_string.assert_called_once()
        sent_msg = json.loads(self.mock_socket.send_string.call_args[0][0])
        self.assertEqual(sent_msg["type"], "status_request")

    def test_execute_test_sequence_normal(self):
        """Test execution of the normal test sequence."""
        # Set up mock responses for each command in the sequence
        self.mock_socket.recv_string.side_effect = [
            json.dumps({"status": "ready"}),
            json.dumps({"status": "executed", "command": "maintain_course"}),
            json.dumps({"status": "ready"}),
            json.dumps({"status": "executed", "command": "increase_velocity"}),
            json.dumps({"status": "ready"}),
            json.dumps({"status": "executed", "command": "maintain_course"}),
            json.dumps({"status": "ready"}),
            json.dumps({"status": "executed", "command": "adjust_trajectory"})
        ]

        # Execute test sequence with a minimal delay
        with patch('time.sleep') as mock_sleep:
            result = self.client.execute_test_sequence("normal", 0.01)

        # Check the result
        self.assertTrue(result)

        # Verify the correct number of commands were sent
        self.assertEqual(self.mock_socket.send_string.call_count, 8)  # 2 per command (request + command)

    def test_execute_test_sequence_invalid(self):
        """Test execution of an invalid test sequence."""
        # Execute non-existent test sequence
        result = self.client.execute_test_sequence("nonexistent", 0.01)

        # Check the result
        self.assertFalse(result)

        # Verify no commands were sent
        self.mock_socket.send_string.assert_not_called()

    def test_random_command_test(self):
        """Test the random command test functionality."""
        # Set up mock responses
        responses = []
        for _ in range(6):  # 3 commands × 2 calls per command
            responses.extend([
                json.dumps({"status": "ready"}),
                json.dumps({"status": "executed"})
            ])
        self.mock_socket.recv_string.side_effect = responses

        # Execute random command test with minimal delay and fixed count
        with patch('time.sleep'), patch('random.choice', return_value="maintain_course"):
            self.client.random_command_test(count=3, delay=0.01)

        # Verify the correct number of commands were sent
        self.assertEqual(self.mock_socket.send_string.call_count, 6)  # 2 per command (request + command)


if __name__ == '__main__':
    unittest.main()