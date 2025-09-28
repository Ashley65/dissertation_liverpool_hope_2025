import numpy as np
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional



class MissionPhaseAwareness:
    """
    Integrated system for mission phase awareness that enables phase-specific behaviours and adaptations.

    Components:
    - PhaseRecognitionModule: Determines the current mission phase
    - PolicyAdaptationMechanism: Injects phase information and modifies rewards
    - ThresholdManagementSubsystem: Adjusts anomaly thresholds and emergency triggers
    - ExperienceSegmentation: Tags replay experiences with phase data
    """

    def __init__(self):
        # Phase recognition module
        self.current_phase = "TRANSIT"  # Default phase
        self.phase_history = []
        self.phase_confidence = {
            "LAUNCH": 0.0,
            "TRANSIT": 1.0,
            "EXPLORATION": 0.0,
            "EMERGENCY": 0.0
        }
        self.phase_conditions = {
            "LAUNCH": self._check_launch_conditions,
            "TRANSIT": self._check_transit_conditions,
            "EXPLORATION": self._check_exploration_conditions,
            "EMERGENCY": self._check_emergency_conditions
        }

        # Policy adaptation mechanism
        self.phase_policies = {
            "LAUNCH": {
                "reward_modifiers": {"fuel_efficiency": 0.5, "speed": 1.5},
                "action_preferences": {"increase_velocity": 1.5}
            },
            "TRANSIT": {
                "reward_modifiers": {"fuel_efficiency": 1.0, "speed": 1.0},
                "action_preferences": {"maintain_course": 1.2}
            },
            "EXPLORATION": {
                "reward_modifiers": {"data_collection": 2.0, "safety": 1.5},
                "action_preferences": {"investigate_anomaly": 1.5}
            },
            "EMERGENCY": {
                "reward_modifiers": {"safety": 3.0, "fuel_efficiency": 0.2},
                "action_preferences": {"emergency_protocol": 2.0}
            }
        }

        # Threshold management
        self.base_thresholds = {
            "anomaly_detection": 0.7,
            "emergency_trigger": 0.85,
            "fuel_critical": 0.15,
            "velocity_warning": 1.5
        }
        self.phase_threshold_modifiers = {
            "LAUNCH": {"anomaly_detection": 0.9, "emergency_trigger": 0.95},
            "TRANSIT": {"anomaly_detection": 0.7, "emergency_trigger": 0.85},
            "EXPLORATION": {"anomaly_detection": 0.5, "emergency_trigger": 0.75},
            "EMERGENCY": {"anomaly_detection": 0.3, "emergency_trigger": 0.6}
        }
        self.current_thresholds = self.base_thresholds.copy()

        # Experience segmentation
        self.experience_buffer = []
        self.max_experiences = 1000
        self.phase_statistics = defaultdict(lambda: {
            "duration": 0,
            "fuel_consumption": 0,
            "velocity_avg": 0,
            "success_rate": 0,
            "action_counts": defaultdict(int)
        })

        logging.info("Mission Phase Awareness system initialized")

    def update(self, craft_status, optimization_data=None):
        """
        Update the phase management system based on current spacecraft status.

        Args:
            craft_status: Current status of the spacecraft
            optimization_data: Optional data from trajectory optimisation

        Returns:
            Dict containing phase information and policy recommendations
        """
        # Determine the current phase
        previous_phase = self.current_phase
        self.update_phase(craft_status, optimization_data)

        # Update thresholds based on the current phase
        self.update_thresholds()

        # Record experience
        if optimization_data:
            self.record_experience(craft_status, optimization_data)

        # Prepare phase data for return
        phase_data = {
            "current_phase": self.current_phase,
            "phase_changed": previous_phase != self.current_phase,
            "phase_confidence": self.phase_confidence,
            "thresholds": self.current_thresholds,
            "policy": self.phase_policies[self.current_phase],
        }

        # Log phase change if it occurred
        if phase_data["phase_changed"]:
            logging.info(f"Mission phase changed: {previous_phase} -> {self.current_phase}")
            self.phase_history.append({
                "from": previous_phase,
                "to": self.current_phase,
                "timestamp": craft_status.get("timestamp", 0),
                "position": craft_status.get("position", [0, 0, 0]),
                "fuel": craft_status.get("fuel_level", 0)
            })

        return phase_data

    def update_phase(self, craft_status, optimization_data=None):
        """
        Update the current mission phase based on spacecraft status.

        Args:
            craft_status: Current status of the spacecraft
            optimization_data: Optional data from trajectory optimisation
        """
        # Calculate confidence for each phase
        confidences = {}
        for phase, condition_checker in self.phase_conditions.items():
            confidences[phase] = condition_checker(craft_status, optimization_data)

        # Apply hysteresis to prevent rapid phase switching
        # If we are >80% confident about a new phase or <20% confident about current phase
        current_conf = confidences[self.current_phase]
        most_likely_phase = max(confidences, key=confidences.get)
        max_conf = confidences[most_likely_phase]

        if (most_likely_phase != self.current_phase and max_conf > 0.8) or current_conf < 0.2:
            self.current_phase = most_likely_phase

        # Update confidence values
        self.phase_confidence = confidences

    def _check_launch_conditions(self, status, opt_data=None):
        """Check if conditions match launch phase."""
        mission_time = status.get("mission_time", 0)
        distance_from_origin = np.linalg.norm(np.array(status.get("position", [0, 0, 0])))
        velocity = np.linalg.norm(np.array(status.get("velocity", [0, 0, 0])))

        # Launch phase confidence calculation
        if mission_time < 3600:  # First hour of mission
            time_factor = 1.0 - (mission_time / 3600)
        else:
            time_factor = 0

        # Distance factor - launch phase if close to origin
        if distance_from_origin < 1e6:  # Within 1000 km
            distance_factor = 1.0 - (distance_from_origin / 1e6)
        else:
            distance_factor = 0

        # Velocity factor - during launch, velocity typically increases
        velocity_factor = min(velocity / 10000, 1.0) if velocity > 0 else 0

        # Compute overall confidence
        confidence = 0.4 * time_factor + 0.3 * distance_factor + 0.3 * velocity_factor
        return confidence

    def _check_transit_conditions(self, status, opt_data=None):
        """Check if conditions match transit phase."""
        position = np.array(status.get("position", [0, 0, 0]))
        velocity = np.array(status.get("velocity", [0, 0, 0]))

        # Get distances to origin and destination
        distance_from_origin = np.linalg.norm(position)

        # Handle case when target_position is None
        target_position = status.get("target_position")
        if target_position is None:
            # If no target, assume high confidence in transit phase
            # since we're away from origin but no specific destination
            destination_factor = 1.0 if distance_from_origin > 1e6 else 0.5
        else:
            target_position = np.array(target_position)
            distance_to_target = np.linalg.norm(position - target_position)
            if distance_to_target > 1e6:  # Not close to destination yet
                destination_factor = 1.0
            else:
                destination_factor = distance_to_target / 1e6

        # Distance from origin factor
        if distance_from_origin > 1e6:  # Beyond 1000 km from origin
            origin_factor = 1.0
        else:
            origin_factor = distance_from_origin / 1e6

        # Velocity stability factor
        velocity_mag = np.linalg.norm(velocity)
        if 1000 < velocity_mag < 30000:  # Reasonable transit velocity
            velocity_factor = 1.0
        else:
            velocity_factor = 0.5

        # Compute overall confidence
        confidence = 0.3 * origin_factor + 0.4 * destination_factor + 0.3 * velocity_factor
        return confidence

    def _check_exploration_conditions(self, status, opt_data=None):
        """Check if conditions match exploration phase."""
        position = np.array(status.get("position", [0, 0, 0]))
        velocity = np.array(status.get("velocity", [0, 0, 0]))
        target_position = status.get("target_position")

        # Distance factor - handle case when target_position is None
        if target_position is None:
            distance_factor = 0.0  # No exploration without target
        else:
            target_position = np.array(target_position)
            distance_to_target = np.linalg.norm(position - target_position)
            # Within 1000 km of target
            if distance_to_target < 1e6:
                distance_factor = 1.0 - (distance_to_target / 1e6)
            else:
                distance_factor = 0.0

        # Velocity factor - lower velocity during exploration
        velocity_mag = np.linalg.norm(velocity)
        if velocity_mag < 1000:
            velocity_factor = 1.0 - (velocity_mag / 1000)
        else:
            velocity_factor = 0.0

        # Command factor - check if exploration commands are active
        command_factor = 0.0
        current_command = status.get("current_command", "")
        if current_command in ["investigate_anomaly", "collect_samples"]:
            command_factor = 1.0

        # Compute overall confidence
        confidence = 0.4 * distance_factor + 0.3 * velocity_factor + 0.3 * command_factor
        return confidence

    def _check_emergency_conditions(self, status, opt_data=None):
        """Check if conditions match emergency phase."""
        fuel_level = status.get("fuel_level", 1.0)
        collision_warning = status.get("collision_warning", False)
        system_failures = status.get("system_failures", [])
        current_command = status.get("current_command", "")

        # Fuel factor
        fuel_factor = 0.0
        if fuel_level < self.current_thresholds["fuel_critical"]:
            fuel_factor = 1.0 - (fuel_level / self.current_thresholds["fuel_critical"])

        # Collision factor
        collision_factor = 1.0 if collision_warning else 0.0

        # System failure factor
        failure_factor = min(len(system_failures) / 3, 1.0)

        # Command factor
        command_factor = 1.0 if current_command == "emergency_protocol" else 0.0

        # Combined emergency confidence
        factors = [fuel_factor, collision_factor, failure_factor, command_factor]
        confidence = max(factors)  # Any critical factor can trigger emergency

        return confidence

    def update_thresholds(self):
        """Update operational thresholds based on the current phase."""
        # Start with base thresholds
        self.current_thresholds = self.base_thresholds.copy()

        # Apply phase-specific modifiers
        if self.current_phase in self.phase_threshold_modifiers:
            for key, modifier in self.phase_threshold_modifiers[self.current_phase].items():
                if key in self.current_thresholds:
                    self.current_thresholds[key] = modifier

    def record_experience(self, status, optimization_data):
        """
        Record experience data for learning and adaptation.

        Args:
            status: Current spacecraft status
            optimization_data: Data from the trajectory optimiser
        """
        # Create experience record with phase tag
        experience = {
            "phase": self.current_phase,
            "timestamp": status.get("timestamp", 0),
            "position": status.get("position", [0, 0, 0]),
            "velocity": status.get("velocity", [0, 0, 0]),
            "fuel_level": status.get("fuel_level", 0),
            "action": status.get("current_command", ""),
            "optimization": optimization_data
        }

        # Add to experience it buffer
        self.experience_buffer.append(experience)

        # Limit buffer size
        if len(self.experience_buffer) > self.max_experiences:
            self.experience_buffer.pop(0)

        # Update phase statistics
        phase_stats = self.phase_statistics[self.current_phase]
        phase_stats["duration"] += 1  # Count update cycles

        # Track fuel consumption
        if len(self.experience_buffer) >= 2:
            prev_fuel = self.experience_buffer[-2].get("fuel_level", 0)
            curr_fuel = experience.get("fuel_level", 0)
            if prev_fuel > curr_fuel:  # Prevent negative consumption from refuelling
                phase_stats["fuel_consumption"] += (prev_fuel - curr_fuel)

        # Track velocity
        velocity_mag = np.linalg.norm(np.array(experience.get("velocity", [0, 0, 0])))
        phase_stats["velocity_avg"] = (phase_stats["velocity_avg"] * (phase_stats["duration"] - 1) + velocity_mag) / \
                                      phase_stats["duration"]

        # Track actions
        action = experience.get("action", "")
        if action:
            phase_stats["action_counts"][action] += 1

    def get_policy_recommendations(self):
        """
        Get policy recommendations based on the current phase.

        Returns:
            Dict containing policy recommendations
        """
        return self.phase_policies[self.current_phase]

    def get_phase_statistics(self):
        """
        Get statistics for each mission phase.

        Returns:
            Dict containing statistics for each phase
        """
        return dict(self.phase_statistics)



    def get_available_phases(self):
        """Returns the list of available mission phases that the system can handle."""
        # Default phases that the system recognizes
        standard_phases = ["TRANSIT", "LAUNCH", "EXPLORATION", "CRITICAL", "EMERGENCY"]

        # You could also extract this from the phase_confidence dictionary keys
        current_phases = list(self.phase_confidence.keys())

        # Combine standard phases with any dynamic phases that might be in the confidence map
        available_phases = list(set(standard_phases + current_phases))

        return available_phases




    def get_experiences_by_phase(self, phase=None, count=10):
        """
        Get recent experiences from a specific phase.

        Args:
            phase: The phase to get experiences from (None for all phases)
            count: Maximum number of experiences to return

        Returns:
            List of experiences
        """
        if phase is None:
            return self.experience_buffer[-count:]

        # Filter by phase
        phase_experiences = [exp for exp in self.experience_buffer if exp["phase"] == phase]
        return phase_experiences[-count:]

    def integrate_astro_data(self, astro_data_manager, status):
        """
        Update phase determination using astronomical data.

        Args:
            astro_data_manager: Instance of AstroDataManager
            status: Current spacecraft status
        """
        # Get current position
        position = np.array(status.get("position", [0, 0, 0]))

        # Find nearest celestial bodies
        nearest_bodies = []
        for body_id, body_data in astro_data_manager.celestial_objects.items():
            body_pos = np.array(body_data.get("position", [0, 0, 0]))
            distance = np.linalg.norm(position - body_pos)
            nearest_bodies.append((body_id, body_data, distance))

        # Sort by distance
        nearest_bodies.sort(key=lambda x: x[2])

        # Adjust phase detection based on celestial context
        if nearest_bodies:
            nearest_id, nearest_data, nearest_distance = nearest_bodies[0]
            body_type = nearest_data.get("type", "").lower()

            # Update phase confidence based on astronomical context
            if body_type == "planet" and nearest_distance < 5e5:
                # Increase exploration confidence when near a planet
                self.phase_confidence["EXPLORATION"] += 0.2

            # Check for hazards that might trigger emergency phase
            hazards = astro_data_manager.get_hazards_near_path(
                "spacecraft", nearest_id,
                safety_distance=0.1
            )

            if hazards:
                # Increase emergency confidence when hazards detected
                self.phase_confidence["EMERGENCY"] += 0.3

    def adapt_optimizer_params(self, optimizer_config):
        """
        Adapt optimiser parameters based on the current phase.

        Args:
            optimizer_config: Current optimiser configuration

        Returns:
            Updated optimiser configuration
        """
        # Create a copy to avoid modifying the original
        config = optimizer_config.copy()

        # Apply phase-specific adaptations
        if self.current_phase == "LAUNCH":
            config["speed_weight"] = 1.5
            config["fuel_weight"] = 0.7
            config["safety_margin"] = 1.2

        elif self.current_phase == "TRANSIT":
            config["speed_weight"] = 1.0
            config["fuel_weight"] = 1.2
            config["safety_margin"] = 1.0

        elif self.current_phase == "EXPLORATION":
            config["speed_weight"] = 0.5
            config["fuel_weight"] = 0.8
            config["safety_margin"] = 1.5
            config["precision_factor"] = 1.5

        elif self.current_phase == "EMERGENCY":
            config["speed_weight"] = 0.8
            config["fuel_weight"] = 0.5
            config["safety_margin"] = 2.0
            config["emergency_route"] = True

        return config