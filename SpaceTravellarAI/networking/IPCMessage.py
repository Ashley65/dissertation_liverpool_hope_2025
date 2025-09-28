import zmq
import json
import time
import logging
from threading import Lock
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class IPCMessage:
    """Structure for IPC messages"""
    message_type: str
    payload: Dict[str, Any]
    timestamp: float
    sequence_id: int


class SpacecraftInterface:
    """Handles ZMQ communication with spacecraft firmware"""

    def __init__(self, connect_address="tcp://localhost:5555"):
        # Initialize ZMQ context and sockets
        self.context = zmq.Context()
        self.command_socket = self.context.socket(zmq.REQ)
        self.telemetry_socket = self.context.socket(zmq.SUB)

        # Connect sockets
        self.command_socket.connect(connect_address)
        self.telemetry_socket.connect(f"{connect_address[:-1]}6")  # Telemetry on next port
        self.telemetry_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Synchronization
        self._lock = Lock()
        self._sequence = 0
        self._last_ping = time.time()
        self.ping_interval = 1.0  # 1 second ping interval

        logging.info("SpacecraftInterface initialized")

    def send_command(self, action: str, parameters: Dict[str, Any]) -> bool:
        """Send command to spacecraft firmware with retry logic"""
        with self._lock:
            message = IPCMessage(
                message_type="command",
                payload={"action": action, "parameters": parameters},
                timestamp=time.time(),
                sequence_id=self._sequence
            )

            try:
                # Send with timeout
                self.command_socket.send_json(message.__dict__, flags=zmq.NOBLOCK)

                # Wait for acknowledgment with timeout
                if self.command_socket.poll(timeout=1000):  # 1 second timeout
                    response = self.command_socket.recv_json()
                    if response.get("status") == "acknowledged":
                        self._sequence += 1
                        logging.debug(f"Command {action} acknowledged: seq={self._sequence}")
                        return True

                logging.warning(f"Command {action} not acknowledged")
                return False

            except zmq.ZMQError as e:
                logging.error(f"ZMQ error sending command: {e}")
                return False

    def receive_telemetry(self, timeout_ms: int = 100) -> Optional[Dict[str, Any]]:
        """Receive telemetry data with timeout"""
        try:
            if self.telemetry_socket.poll(timeout_ms):
                data = self.telemetry_socket.recv_json()
                return data
            return None

        except zmq.ZMQError as e:
            logging.error(f"ZMQ error receiving telemetry: {e}")
            return None

    def check_connection(self) -> bool:
        """Verify connection is alive with ping"""
        current_time = time.time()

        if current_time - self._last_ping >= self.ping_interval:
            ping_message = IPCMessage(
                message_type="ping",
                payload={},
                timestamp=current_time,
                sequence_id=self._sequence
            )

            try:
                self.command_socket.send_json(ping_message.__dict__, flags=zmq.NOBLOCK)
                if self.command_socket.poll(timeout=500):  # 500ms timeout
                    response = self.command_socket.recv_json()
                    self._last_ping = current_time
                    return True
                return False

            except zmq.ZMQError:
                return False

        return True

    def close(self):
        """Clean up ZMQ resources"""
        self.command_socket.close()
        self.telemetry_socket.close()
        self.context.term()