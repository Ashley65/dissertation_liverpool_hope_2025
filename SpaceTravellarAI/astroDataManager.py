import json
import numpy as np
import logging
from typing import Dict, List, Optional, Any
import os

from numpy import floating


class AstroDataManager:
    """
    Manager a dynamic astronomical data received from mission control.
    Provides methods to process, store, and retrieve data.
    """
    def __init__(self, data_dir="./data"):
        self.data_directory = data_dir
        self.celestial_objects = {} # Dictionary to store celestial objects data
        self.mission_targets = {}  # Dictionary to store mission-specific targets
        self.navigation_waypoints = {} # store computed waypoints between locations

        # Create the data directory if it does not exist
        if not os.path.exists(data_dir):
            os.makedirs(self.data_directory, exist_ok=True)

        self._load_cached_data()
        logging.info(f"AstroDataManager initialised with {len(self.celestial_objects)} celestial bodies")


    def _load_cached_data(self):
        """ Load the celestial objects data from the cache directory."""

        try:
            cache_file = os.path.join(self.data_directory, "celestial_data.json")
            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    self.celestial_objects = data.get("celestial_bodies", {})
                    self.mission_targets = data.get("mission_targets", {})
                logging.info(f"Loaded {len(self.celestial_objects)} celestial bodies from cache")
        except Exception as e:
            logging.error(f"Failed to load cached data: {e}")

    def _save_cached_data(self):
        """ save celestial body data to the cache directory."""
        try:
            cache_file = os.path.join(self.data_directory, "celestial_data.json")

            # Create a copy of the data to modify for serialization
            serializable_celestial_objects = {}
            for body_id, body_data in self.celestial_objects.items():
                serializable_body = {}
                for key, value in body_data.items():
                    # Convert numpy arrays to lists for JSON serialization
                    if isinstance(value, np.ndarray):
                        serializable_body[key] = value.tolist()
                    else:
                        serializable_body[key] = value
                serializable_celestial_objects[body_id] = serializable_body

            # Similarly for mission_targets
            serializable_mission_targets = {}
            for mission_id, target_data in self.mission_targets.items():
                serializable_target = {}
                for key, value in target_data.items():
                    if isinstance(value, np.ndarray):
                        serializable_target[key] = value.tolist()
                    else:
                        serializable_target[key] = value
                serializable_mission_targets[mission_id] = serializable_target

            data = {
                "celestial_bodies": serializable_celestial_objects,
                "mission_targets": serializable_mission_targets
            }

            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
            logging.info(f"Saved data to cache: {cache_file}")

        except Exception as e:
            logging.error(f"Failed to save data to cache: {e}")

    def update_celestial_data(self, data_packet):
        """
        Update the celestial body data from a mission data packet.
        :param data_packet: The dictionary containing the data packet

        """
        if not isinstance(data_packet, dict):
            logging.error(f"Invalid data packet type: {type(data_packet)}")
            return False

        bodies = data_packet.get("celestial_bodies", {})
        if not bodies:
            logging.error(f"No celestial bodies data found in packet")
            return False

        # Update the celestial objects data with new information
        for body_id, body_data in bodies.items():
            if body_id in self.celestial_objects:
                # Update the existing body data
                self.celestial_objects[body_id].update(body_data)
                logging.debug(f"Updated celestial object {body_id} data")
            else:
                # Add a new celestial body
                self.celestial_objects[body_id] = body_data
                logging.debug(f"Added new celestial object {body_id}")

        # Save the updated data to cache
        self._save_cached_data()
        return True

    def get_body_data(self, body_id: str) -> Dict:
        """Get data for a specific celestial body."""
        if body_id in self.celestial_objects:
            return self.celestial_objects[body_id]
        else:
            logging.warning(f"Celestial body {body_id} not found in data")
            return {}  # Return empty dict instead of None to prevent attribute errors



    def get_bodies_by_type(self, body_type) -> dict[Any, Any]:
        """
                Get all celestial bodies of a specific type.

                Args:
                    body_type: Type of celestial body (e.g. 'asteroid', 'planet')

                Returns:
                    Dictionary of bodies matching the specified type
                """
        return {k: v for k, v in self.celestial_objects.items()
                if v.get('type', '').lower() == body_type.lower()}

    def add_celestial_object(self, name: str, data: dict):
        """
        Add or update a celestial object in the database.

        Args:
            name (str): Name of the celestial object
            data (dict): Dictionary containing object properties
                Required keys: position, mass, radius
                Optional: atmosphere, landing_sites
        """
        if not all(key in data for key in ['position', 'mass', 'radius']):
            raise ValueError("Missing required celestial object properties")

        # Convert position to numpy array if it isn't already
        if not isinstance(data['position'], np.ndarray):
            data['position'] = np.array(data['position'])

        self.celestial_objects[name] = data
        return True

    def get_object_data(self, name: str) -> dict:
        """
        Retrieve data for a specific celestial object.

        Args:
            name (str): Name of the celestial object

        Returns:
            dict: Object data or empty dict if not found
        """
        return self.celestial_objects.get(name, {})

    def update_object_position(self, name: str, new_position: np.ndarray):
        """
        Update the position of a celestial object.

        Args:
            name (str): Name of the celestial object
            new_position (np.ndarray): New position vector
        """
        if name in self.celestial_objects:
            self.celestial_objects[name]['position'] = new_position
            return True
        return False

    def calculate_gravitational_force(self, pos1: np.ndarray, mass1: float,
                                      pos2: np.ndarray, mass2: float) -> np.ndarray:
        """
        Calculate gravitational force between two objects.

        Args:
            pos1 (np.ndarray): Position vector of first object
            mass1 (float): Mass of first object
            pos2 (np.ndarray): Position vector of second object
            mass2 (float): Mass of second object

        Returns:
            np.ndarray: Gravitational force vector
        """
        r = pos2 - pos1
        r_mag = np.linalg.norm(r)
        r_hat = r / r_mag
        force_mag = self.astronomical_constants['G'] * mass1 * mass2 / (r_mag ** 2)
        return force_mag * r_hat

    def add_body_data(self, body_id: str, body_data: Dict[str, Any]) -> bool:
        """
        Add or update data for a specific celestial body.

        Args:
            body_id: The unique identifier of the celestial body
            body_data: The dictionary containing the body data

        Returns:
            bool: True if successful, False otherwise
        """
        if not isinstance(body_data, dict):
            logging.error(f"Invalid body data type: {type(body_data)}")
            return False

        if body_id in self.celestial_objects:
            # Update the existing body data
            self.celestial_objects[body_id].update(body_data)
            logging.debug(f"Updated celestial object {body_id} data")
        else:
            # Add a new celestial body
            self.celestial_objects[body_id] = body_data
            logging.debug(f"Added new celestial object {body_id}")

        # Save the updated data to cache
        self._save_cached_data()
        return True


    def calculate_distance(self, origin_id, destination_id) -> Optional[floating[Any]]:
        """
        Calculate the distance between two celestial bodies.
        :param origin_id: The unique identifier of the origin body
        :param destination_id: The unique identifier of the destination body
        :return: Distance as float, or None if bodies are not found
        """
        origin = self.get_body_data(origin_id)
        destination = self.get_body_data(destination_id)

        if not origin or not destination:
            return None

        origin_pos = origin.get("position", None)
        destination_pos = destination.get("position", None)

        if not origin_pos or not destination_pos:
            return None
        # convert to numpy arrays if they are not already in that format
        origin_pos_arr = np.array(origin_pos) if not isinstance(origin_pos, np.ndarray) else origin_pos
        destination_pos_arr = np.array(destination_pos) if not isinstance(destination_pos, np.ndarray) else destination_pos

        distance = np.linalg.norm(destination_pos_arr - origin_pos_arr)
        return distance

    def add_mission_target(self, mission_id, target_data):
        """
        Add a mission-specific target to the data manager with relevant information.
        :param mission_id: Identifier for the mission
        :param target_data: Dictionary containing target data
        """

        self.mission_targets[mission_id] = target_data
        logging.info(f"Added mission target {mission_id}")

    def get_mission_target(self, mission_id):
        """
        Retrieve the target data for a specific mission.
        :param mission_id: Identifier for the mission
        :return: Dictionary containing the target data
        """
        return self.mission_targets.get(mission_id, None)

    # note needs to make this method dynamic when computing waypoints by allowing the AI to increase or decrease the number of waypoints depending on the mission requirements
    def compute_navigation_waypoints(self, start_id, end_id, num_waypoints=5, mission_phase=None):
        """
        compute a series of waypoints between two celestial bodies.
        :param start_id: Start body identifier
        :param end_id: destination body identifier
        :param num_waypoints: number of waypoints generated between the start and end body
        :param mission_phase: Optional mission phase for specific waypoint calculations

        :return: A list of waypoint positions pr None if the computation fails
        """

        # Phase-specific waypoint adjustments
        if mission_phase == "EXPLORATION":
            # More detailed path during exploration
            num_waypoints = max(num_waypoints, 8)
        elif mission_phase == "EMERGENCY":
            # Simplified, direct path during emergency
            num_waypoints = min(num_waypoints, 3)
        elif mission_phase == "LAUNCH":
            # Precise initial trajectory
            num_waypoints = max(num_waypoints, 6)

        start_body = self.get_body_data(start_id)
        end_body = self.get_body_data(end_id)

        if not start_body or not end_body:
            logging.error(f"Failed to compute waypoints: Missing celestial body data")
            return None

        start_pos = np.array(start_body.get("position", None))
        end_pos = np.array(end_body.get("position", None))

        # Fix: Check if positions are None or empty using size check
        if start_pos is None or end_pos is None or start_pos.size == 0 or end_pos.size == 0:
            logging.error(f"Failed to compute waypoints: Missing position data")
            return None

        # Compute the intermediate waypoints
        # In a real system, this would use orbital mechanics
        waypoints = []
        for i in range(1, num_waypoints + 1):
            alpha = i / (num_waypoints + 1)
            waypoint = start_pos + alpha * (end_pos - start_pos)
            waypoints.append(waypoint.tolist())

        # Store for future reference
        key = f"{start_id}_to_{end_id}"
        self.navigation_waypoints[key] = waypoints

        return waypoints


    def get_hazards_near_path(self, start_id, end_id, safety_distance=0.1, mission_phase=None):
        """
        Identify potential environmental hazards along a path between two bodies.
        :param start_id: Start body identifier
        :param end_id: Destination body identifier
        :param safety_distance:  Minimum safe distance to other bodies
        :param mission_phase: Optional mission phase for specific hazard calculations
        :return: List of hazardous body IDs
        """

        # Adjust safety distance based on mission phase
        if mission_phase == "EXPLORATION":
            safety_distance *= 1.5  # More cautious during exploration
        elif mission_phase == "EMERGENCY":
            safety_distance *= 2.0  # Maximum caution during emergency
        elif mission_phase == "LAUNCH":
            safety_distance *= 1.2  # Extra caution during launch

        # Get or compute the waypoints between the start and end bodies
        key = f"{start_id}_to_{end_id}"
        if key not in self.navigation_waypoints:
            self.compute_navigation_waypoints(start_id, end_id)

        waypoints = self.navigation_waypoints.get(key, [])
        if not waypoints:
            logging.error(f"No waypoints found for {start_id} to {end_id}")
            return []

        # Check all bodies against each waypoint
        hazardous  = []
        for body_id, body_data in self.celestial_objects.items():
            # Skip the start and end bodies
            if body_id in [start_id, end_id]:
                continue

            # Only consider certain types of bodies as hazards
            body_type = body_data.get("type", "").lower()
            if body_type not in ['asteroid', 'radiation_belt', 'debris_field', 'planet']:
                continue

            body_pos = body_data.get("position")
            if not body_pos:
                continue

            body_pos = np.array(body_pos)

            # Check the distance for each waypoint
            for waypoint in waypoints:
                distance = np.linalg.norm(waypoint - body_pos)
                if distance < safety_distance:
                    hazardous.append({
                        "body_id": body_id,
                        "type": body_type,
                        "distance": float(distance)
                    })
                    break
        return hazardous


