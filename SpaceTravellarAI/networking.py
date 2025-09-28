
import time


import zmq


import json
import re
import numpy as np
from collections import defaultdict
import logging
from typing import Dict, List, Any, Optional
from parameterised import ParameterisedCommand

# Replace spaCy import with NLTK
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk

    NLTK_AVAILABLE = True
    logging.info("NLTK is available and will be used for command parsing")

    # Download necessary NLTK data if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logging.info("Downloading NLTK punkt tokenizer")
        nltk.download('punkt')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        logging.info("Downloading NLTK POS tagger")
        nltk.download('averaged_perceptron_tagger')
    try:
        nltk.data.find('chunkers/maxent_ne_chunker')
    except LookupError:
        logging.info("Downloading NLTK NE chunker")
        nltk.download('maxent_ne_chunker')
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        logging.info("Downloading NLTK words corpus")
        nltk.download('words')

except ImportError as e:
    NLTK_AVAILABLE = False
    logging.warning(f"NLTK import failed: {e}. Using rule-based parsing instead")

class CommandParser:
    """Paress complex commands into simpler actions for the AI Captain to execute. Using a natural language processing model."""

    def __init__(self, model_name="en_core_web_sm"):
        """Initialise the CommandParser class."""

        self.model_name = model_name
        self.nlp = None

        # Initialise the NLP model if available
        if NLTK_AVAILABLE:
            self.nlp = True  # Use True as a flag for NLTK availability
            logging.info("NLTK is ready for command parsing")



        # command keywords and their corresponding actions
        self.command_categories = {
            'navigation': ['proceed', 'navigate', 'trajectory', 'course', 'orbit', 'fly', 'travel', 'return'],
            'sampling': ['collect', 'sample', 'gather', 'extract', 'analyze', 'survey'],
            'communication': ['communicate', 'transmit', 'signal', 'broadcast', 'link', 'contact'],
            'avoidance': ['avoid', 'evade', 'bypass', 'steer', 'clear'],
            'observation': ['observe', 'monitor', 'scan', 'watch', 'track'],
            'landing': ['land', 'descent', 'touchdown', 'approach'],
            'docking': ['dock', 'rendezvous', 'attach', 'connect']
        }

        # Standard task templates that match the AI Captain's abilities
        self.task_templates = {
            "navigation": {
                "task": "Compute trajectory",
                "parameters": {"target": "", "fuel_limit": "optimal"}
            },
            "monitoring": {
                "task": "Monitor hazards",
                "parameters": {"avoidance": "high-priority", "objects": []}
            },
            "communication": {
                "task": "Maintain communications",
                "parameters": {"target": "Earth", "interval": "continuous"}
            },
            "landing": {
                "task": "Land on target",
                "parameters": {"method": "controlled descent"}
            },
            "sampling": {
                "task": "Collect samples",
                "parameters": {"method": "robotic arm"}
            },
            "return": {
                "task": "Return to base",
                "parameters": {"target": "mother ship", "dock_safely": True}
            }
        }
        # Action mapping for AI Captain
        self.action_mapping = {
            "Compute trajectory": ["adjust_trajectory"],
            "Monitor hazards": ["scan_area", "adjust_trajectory"],
            "Maintain communications": ["maintain_course"],
            "Land on target": ["decrease_velocity"],
            "Collect samples": ["investigate_anomaly"],
            "Return to base": ["adjust_trajectory", "return_to_base"]
        }

    def parse_command(self, command_text):
        """
        Parse a complex command into structured mission details.

        Args:
            command_text: The command text to parse

        Returns:
            A structured mission specification
        """
        # Initialize mission structure
        mission = {
            "mission": self._determine_mission_type(command_text),
            "destination": self._extract_destination(command_text),
            "tasks": [],
            "priority_level": self._determine_priority(command_text)
        }

        if self.nlp:
            # Use spaCy for parsing if available
            mission = self._parse_with_nlp(command_text, mission)
        else:
            # Fall back to rule-based parsing
            mission = self._parse_with_rules(command_text, mission)

        return mission

    def _parse_with_nlp(self, command_text, mission):
        """Parse command using NLTK."""
        # Tokenize and tag the text
        tokens = word_tokenize(command_text)
        tagged_tokens = pos_tag(tokens)

        # Extract potential targets and actions
        targets = []
        actions = []

        # Find verbs that match our command categories
        for word, tag in tagged_tokens:
            if tag.startswith('VB'):  # Verb
                verb = word.lower()
                for category, keywords in self.command_categories.items():
                    if verb in keywords:
                        actions.append((category, verb))

        # Extract potential targets (nouns following verbs)
        for i in range(1, len(tagged_tokens)):
            word, tag = tagged_tokens[i]
            prev_word, prev_tag = tagged_tokens[i - 1]

            # If previous word is a preposition and current word is a noun
            if prev_tag.startswith('IN') and tag.startswith('NN'):
                if word not in targets:
                    targets.append(word)

            # If previous word is a verb and current word is a noun
            if prev_tag.startswith('VB') and tag.startswith('NN'):
                if word not in targets:
                    targets.append(word)

        # Process named entities for destinations
        named_entities = ne_chunk(tagged_tokens)
        for chunk in named_entities:
            if hasattr(chunk, 'label') and chunk.label() == 'GPE':  # Location
                entity_name = ' '.join(c[0] for c in chunk)
                if entity_name not in targets:
                    targets.append(entity_name)

        # If we did not find a destination yet, try to use one of the targets
        if not mission["destination"] and targets:
            mission["destination"] = targets[0]

        # Process extracted actions (same as before)
        for category, _ in actions:
            if category == "navigation":
                task = self.task_templates["navigation"].copy()
                task["parameters"]["target"] = mission["destination"]
                mission["tasks"].append(task)

            elif category == "avoidance":
                task = self.task_templates["monitoring"].copy()
                objects = []
                if "gravitational" in command_text and "wells" in command_text:
                    objects.append("gravitational wells")
                if not objects:
                    objects = ["all hazards"]
                task["parameters"]["objects"] = objects
                mission["tasks"].append(task)

            elif category == "sampling":
                task = self.task_templates["sampling"].copy()
                mission["tasks"].append(task)

            elif category == "communication":
                task = self.task_templates["communication"].copy()
                if "mother" in command_text.lower():
                    task["parameters"]["target"] = "mother ship"
                mission["tasks"].append(task)

            elif category == "landing":
                task = self.task_templates["landing"].copy()
                mission["tasks"].append(task)

            elif category == "docking" or category == "navigation" and "return" in command_text.lower():
                task = self.task_templates["return"].copy()
                mission["tasks"].append(task)

        # If no tasks were identified, fall back to rule-based parsing
        if not mission["tasks"]:
            mission = self._parse_with_rules(command_text, mission)

        return mission
    def _parse_with_rules(self, command_text, mission):

        """Parse command using a rule-based method."""
        text = command_text.lower()

        # Extract tasks using template matching
        if "proceed to" in text or "navigate" in text or "go to" in text:
            nav_task = self.task_templates["navigation"].copy()
            nav_task["parameters"]["target"] = mission["destination"]
            mission["tasks"].append(nav_task)
        if "avoid" in text:
            avoid_task = self.task_templates["monitoring"].copy()
            avoid_objects = []
            if "gravitational" in text and "wells" in text:
                avoid_objects.append("gravitational wells")
            if not avoid_objects:
                avoid_objects = ["all hazards"]
            avoid_task["parameters"]["objects"] = avoid_objects
            mission["tasks"].append(avoid_task)

        if "collect" in text or "sample" in text or "gather" in text:
            sample_task = self.task_templates["sampling"].copy()
            mission["tasks"].append(sample_task)

        if "communication" in text or "link" in text or "maintain" in text and "contact" in text:
            comm_task = self.task_templates["communication"].copy()
            if "mother" in text:
                comm_task["parameters"]["target"] = "mother ship"
            mission["tasks"].append(comm_task)

        if "land" in text or "descent" in text:
            land_task = self.task_templates["landing"].copy()
            mission["tasks"].append(land_task)

        if "return" in text or "dock" in text or "back to" in text:
            return_task = self.task_templates["return"].copy()
            mission["tasks"].append(return_task)

        return mission


    @staticmethod
    def _determine_mission_type(text):
        """Determine the general mission type from text."""
        text = text.lower()
        if "collect" in text or "sample" in text or "gather" in text:
            return "Sample Collection"
        elif "survey" in text or "map" in text or "explore" in text:
            return "Exploration"
        elif "return" in text or "dock" in text:
            return "Return Mission"
        else:
            return "General Mission"

    @staticmethod
    def _extract_destination(text):
        """Extract the destination from the text."""
        # Pattern to find celestial body references
        pattern = r'(?:to|towards|at|on|near)\s+([A-Za-z0-9\-]+\s*[A-Za-z0-9\-]*)'
        matches = re.findall(pattern, text)
        if matches:
            return matches[0].strip()
        return ""

    @staticmethod
    def _determine_priority(text):
        """Determine the priority level from the text."""
        text = text.lower()
        if any(word in text for word in ['immediately', 'urgent', 'critical', 'emergency']):
            return "high"
        elif any(word in text for word in ['when possible', 'low priority', 'optional']):
            return "low"
        return "normal"

    def generate_action_sequence(self, mission):
        """
        Convert structured mission data into AI Captain action commands.

        Args:
            mission: Structured mission data

        Returns:
            List of action commands for the AI Captain
        """
        action_sequence = []

        for task in mission.get("tasks", []):
            task_name = task.get("task", "")
            mapped_actions = self.action_mapping.get(task_name, [])
            action_sequence.extend(mapped_actions)

        # Add emergency protocol for high-priority missions
        if mission.get("priority_level") == "high":
            if "emergency_protocol" not in action_sequence:
                action_sequence.insert(0, "emergency_protocol")

        # Remove duplicates while preserving order
        unique_actions = []
        for action in action_sequence:
            if action not in unique_actions:
                unique_actions.append(action)

        return unique_actions

    def summarise_mission(self, mission):
        """Generate a human-readable summary of the mission."""
        return json.dumps(mission, indent=2)

class NetworkCommunication:
    """
    This class is responsible for handling network communication between the AI Captain and other systems
    """

    def __init__(self, server_address="tcp://localhost:5555"):
        """
        Initialise the NetworkCommunication class.
        :param server_address: The address to connect to the server.
        """
        self.server_address = server_address
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.server_address)

    @staticmethod
    def _prepare_status_data(status):
        """Convert NumPy types to Python standard types for JSON serialisation."""
        status_copy = {}
        for key, value in status.items():
            if isinstance(value, np.ndarray):
                status_copy[key] = value.tolist()
            elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                    np.uint8, np.uint16, np.uint32, np.uint64)):
                status_copy[key] = int(value)
            elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                status_copy[key] = float(value)
            elif isinstance(value, (bool, np.bool_)):
                status_copy[key] = bool(value)
            else:
                status_copy[key] = value
        return status_copy

    def send_status(self, status):
        try:
            # Convert NumPy types to Python standard types
            status_copy = self._prepare_status_data(status)

            # Set timeout for send operation
            self.socket.setsockopt(zmq.RCVTIMEO, 3000)  # 3 second timeout

            message = json.dumps({"type": "status_update", "data": status_copy})
            self.socket.send_string(message)

            # Wait for acknowledgement from the server
            try:
                response = self.socket.recv_string()
                logging.info(f"Status Acknowledged: {response}")
                return True
            except zmq.error.Again:
                logging.warning("Status acknowledgement timeout. Will retry next cycle.")
                return False

        except Exception as e:
            logging.error(f"Error sending status: {e}")

            # Attempt to reconnect if the connection is lost
            try:
                self.socket = self.context.socket(zmq.REQ)
                self.socket.connect(self.server_address)
                logging.info("Reconnected to server successfully")
            except:
                logging.error("Failed to reconnect to server")
            return False

    def receive_command(self):
        """Request and receive mission commands from the simulation."""
        try:
            self.socket.send_string(json.dumps({"type": "request_command"}))
            response = self.socket.recv_string()

            command_data = json.loads(response)
            print(f"[AI] Received Command: {command_data}")

            # Process complex command text if present
            if "command_text" in command_data:
                # If the CommandParser is not initialised yet, create one
                if not hasattr(self, 'command_parser'):
                    self.command_parser = CommandParser()

                # If we do not have a parameter handler yet, create one
                if not hasattr(self, 'param_handler'):
                    self.param_handler = ParameterisedCommand()

                # First, try to parse as a parameterised command
                param_command = self.param_handler.parse_command_text(command_data["command_text"])
                if param_command:
                    return {"type": "parameterized_command", "command_data": param_command}

                # If that fails, parse as a complex mission
                mission = self.command_parser.parse_command(command_data["command_text"])
                action_sequence = self.command_parser.generate_action_sequence(mission)

                return {
                    "type": "complex_command",
                    "mission": mission,
                    "actions": action_sequence
                }

            # Handle parameterized commands from server
            elif "command" in command_data and "parameters" in command_data:
                return {
                    "type": "parameterized_command",
                    "command_data": {
                        "command": command_data["command"],
                        "parameters": command_data["parameters"]
                    }
                }

            # Return a simple command if present
            elif "command" in command_data:
                return {"type": "simple_command", "command": command_data["command"]}

            # No valid command found
            return None

        except Exception as e:
            print(f"[AI] Error receiving command: {e}")
            return None


class CommandHandler:
    """
    It processes and prioritises the commands received from the mission control server.
    """

    def __init__(self):
        self.command_queue = []
        self.priority_levels = {
            'emergency_protocol': 10,
            'investigate_anomaly': 8,
            'refuel': 7,
            'adjust_trajectory': 5,
            'increase_velocity': 4,
            'decrease_velocity': 4,
            'maintain_course': 3
        }

        self.last_executed_command = None
        self.execution_cooldown = {
            'emergency_protocol': 5,
            'investigate_anomaly': 3,
            'refuel': 2,
            'adjust_trajectory': 1,
            'increase_velocity': 1,
            'decrease_velocity': 1,
            'maintain_course': 1
        }
        self.command_history = {}  # Command name: last executed step
        self.parm_handler = ParameterisedCommand()

        self.command_outcomes = defaultdict(list)# Command name: list of times executed
        self.max_outcome_history = 100 # store up to 100 outcomes per command
        self.execution_results = {}

    def add_command(self, command, source="mission_control", priority_override=None, parameters=None):
        """
        Add a new command to the queue with appropriate prioritisation.

        Args:
            command: The command to be executed or command dict
            source: Source of the command (mission_control, onboard_system, etc.)
            priority_override: Optional manual priority level
            parameters: Optional command parameters
        """
        # Handle command if it's already a dictionary with parameters
        if isinstance(command, dict) and 'command' in command:
            cmd_name = command['command']
            parameters = command.get('parameters', {})
        else:
            cmd_name = command

        if cmd_name not in self.priority_levels and priority_override is None:
            logging.warning(f"Unknown command: {cmd_name}")
            return False

        priority = priority_override if priority_override is not None else self.priority_levels.get(cmd_name, 0)

        # Increase priority for certain situations
        if source == "onboard_system" and cmd_name == "emergency_protocol":
            priority += 5

        # Check previous outcomes for this command type to adjust priority
        if cmd_name in self.command_outcomes and len(self.command_outcomes[cmd_name]) > 0:
            last_outcome = self.command_outcomes[cmd_name][-1]
            if not last_outcome['success']:
                priority += 1

        self.command_queue.append({
            "command": cmd_name,
            "source": source,
            "priority": priority,
            "parameters": parameters
        })

        # Sort by priority (highest first)
        self.command_queue.sort(key=lambda x: x["priority"], reverse=True)
        return True

    def get_next_command(self, current_step):
        """
        Get the highest priority command that can be executed now.

        Args:
            current_step: Current mission step for cooldown tracking

        Returns:
            Next command to execute or None if no valid commands
        """
        if not self.command_queue:
            return None

        # Try commands in priority order
        for i, cmd_data in enumerate(self.command_queue):
            # Handle mission type items differently
            if cmd_data.get("type") == "mission" and cmd_data.get("actions"):
                if cmd_data["actions"]:
                    next_action = cmd_data["actions"].pop(0)
                    self.command_history[next_action] = current_step
                    cmd_data["execution_attempts"] = cmd_data.get("execution_attempts", 0) + 1

                    # If this is the last action, remove the mission from the queue
                    if not cmd_data["actions"]:
                        self.command_queue.pop(i)

                    return next_action
                else:
                    # Empty action list, remove from the queue
                    self.command_queue.pop(i)
                    continue

            # Handle regular commands
            command = cmd_data.get("command")
            if command:
                # Check if the command is on cooldown
                last_step = self.command_history.get(command, 0)
                cooldown = self.execution_cooldown.get(command, 0)

                if current_step - last_step >= cooldown:
                    # Command can be executed
                    self.command_queue.pop(i)
                    self.command_history[command] = current_step
                    cmd_data["execution_attempts"] = cmd_data.get("execution_attempts", 0) + 1
                    self.last_executed_command = command
                    return command

        # If all commands are on cooldown, return None
        return None

    def record_outcome(self, command, success, details=None):
        """
        Record the outcome of a command execution.

        Args:
            command: The command that was executed
            success: Whether the command executed successfully
            details: Additional details about the execution
        """
        outcome = {
            "success": success,
            "timestamp": time.time(),
            "details": details or {}
        }

        # If this was a retry, increase the attempt count
        attempts = 1
        if command in self.execution_results:
            prev_outcome = self.execution_results[command]
            if not prev_outcome["success"]:
                attempts = prev_outcome.get("attempts", 1) + 1

        outcome["attempts"] = attempts

        # Store the result
        self.execution_results[command] = outcome

        # Make sure command_outcomes[command] is a list before appending
        if not isinstance(self.command_outcomes[command], list):
            self.command_outcomes[command] = []

        # Add to historical outcomes
        self.command_outcomes[command].append(outcome)

        # Trim history if needed
        if len(self.command_outcomes[command]) > self.max_outcome_history:
            self.command_outcomes[command] = self.command_outcomes[command][-self.max_outcome_history:]

        # Log the outcome
        log_level = logging.INFO if success else logging.WARNING
        attempt_str = f" (attempt {attempts})" if attempts > 1 else ""
        log_message = f"Command '{command}'{attempt_str}: {'Succeeded' if success else 'Failed'}"
        logging.log(log_level, log_message)

        # Format detailed information for the log file
        formatted_details = ""
        if details:
            # Extract key details for more readable logging
            reason = details.get("reason", "")
            mission_phase = details.get("mission_phase", "")
            fuel_level = details.get("fuel_level", "")

            # Get sensor readings if available
            sensor_data = ""
            if "sensor_readings" in details:
                sensor_values = []
                for key, value in details["sensor_readings"].items():
                    sensor_values.append(f"{key}={value}")
                if sensor_values:
                    sensor_data = f", Sensors: {', '.join(sensor_values)}"

            # Build the formatted details string
            formatted_details = []
            if reason:
                formatted_details.append(f"Reason: {reason}")
            if mission_phase:
                formatted_details.append(f"Phase: {mission_phase}")
            if fuel_level:
                formatted_details.append(f"Fuel: {fuel_level:.2f}")
            if sensor_data:
                formatted_details.append(sensor_data)

            formatted_details = ", ".join(formatted_details)

        # Log to the separate outcomes file with more detailed information
        with open('outcomes.log', 'a') as outcomes_file:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            status = "SUCCESS" if success else "FAILURE"
            outcomes_file.write(
                f"{timestamp} - {command}{attempt_str} - {status} - {formatted_details}\n"
            )

        return True

    def get_command_success_rate(self, command):
        """
        Get the success rate for a specific command.

        Args:
            command: The command to check

        Returns:
            Success rate as a float between 0.0 and 1.0
        """
        if command not in self.command_outcomes or not self.command_outcomes[command]:
            return None

        outcomes = self.command_outcomes[command]
        return sum(1 for o in outcomes if o["success"]) / len(outcomes)

    def get_command_reliability_report(self):
        """
        Generate a report on command reliability.

        Returns:
            Dictionary with command reliability statistics
        """
        report = {}
        for cmd, outcomes in self.command_outcomes.items():
            if not outcomes:
                continue

            success_count = sum(1 for o in outcomes if o["success"])
            report[cmd] = {
                "success_rate": success_count / len(outcomes),
                "total_executions": len(outcomes),
                "last_execution": outcomes[-1]["timestamp"],
                "last_outcome": "success" if outcomes[-1]["success"] else "failure",
                "average_attempts": sum(o.get("attempts", 1) for o in outcomes) / len(outcomes)
            }

        return report


    def add_mission(self, mission_data, action_sequence):
        """
        Add a new mission with its action sequence to the queue

        Args:
            mission_data: The parsed mission data
            action_sequence: List of actions to execute
        """
        priority = 3 if mission_data.get("priority_level") == "high" else \
            1 if mission_data.get("priority_level") == "low" else 2

        self.command_queue.append({
            "type": "mission",
            "mission": mission_data,
            "actions": action_sequence.copy(),
            "original_actions": action_sequence.copy(),
            "priority": priority,
            "timestamp": time.time()
        })

        # Sort queue by priority (higher number = higher priority)
        self.command_queue.sort(key=lambda x: x.get("priority", 0), reverse=True)

    def clear_queue(self):
        """Clear all pending commands."""
        queue_size = len(self.command_queue)
        self.command_queue = []
        return queue_size