import logging

import numpy as np
from scipy.constants import G



class FuelOptimiser:

    """ optimiser class for the fuel consumption based on the mission phase, spacecraft status, and the trajectory. """

    def __init__(self, astro_data_manager, initial_fuel_level=1.0):
        self.fuel_level = initial_fuel_level
        self.astro_data = astro_data_manager

        # Fuel consumption rates for different phases of the mission and need to add a variable that
        self.consumption_rates = {
            "LAUNCH": 0.05,        # High consumption during launch
            "TRANSIT": 0.02,       # Moderate consumption during transit
            "EXPLORATION": 0.03,   # Higher than transit due to maneuvering
            "CRITICAL": 0.04,      # Higher consumption during critical operations
            "EMERGENCY": 0.01      # Minimal consumption to preserve fuel
        }

        self.action_multipliers ={
            "adjust_trajectory": 1.5,  # Uses more fuel
            "increase_velocity": 2.0,  # Uses significantly more fuel
            "decrease_velocity": 1.2,  # Uses moderate fuel
            "maintain_course": 0.8,  # Uses less fuel
            "investigate_anomaly": 1.1,  # Slight increase in fuel usage
            "emergency_protocol": 0.5,  # Emergency protocols conserve fuel
            "refuel": 0.0  # No fuel consumption during refueling
        }

        self.craft_status = {
            "position": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "mass" : 1000.0, # kg
            "anomaly_detected": False
        }


        self.efficiency_history = []
        self.trajectory_history = []
        self.last_optimisation = None

        # Orbital mechanics parameters
        self.gravity_assist_benefit = 0.3  # 30% fuel savings with gravity assist
        self.orbital_efficiency_factor = 0.8  # 20% more efficient when using proper orbital mecha

    def get_config(self):
        """
        Return the current configuration of the fuel optimiser.

        Returns:
            Dictionary containing the optimiser's configuration
        """
        config = {
            "fuel_level": self.fuel_level,
            "consumption_rates": self.consumption_rates.copy(),
            "action_multipliers": self.action_multipliers.copy(),
            "craft_status": self.craft_status.copy(),
            "gravity_assist_benefit": self.gravity_assist_benefit,
            "orbital_efficiency_factor": self.orbital_efficiency_factor
        }

        if self.last_optimisation:
            config["last_optimisation"] = self.last_optimisation.copy()

        return config

    def update_config(self, config):
        """
        Update the fuel optimiser configuration with new values.

        Args:
            config: Dictionary containing the updated configuration parameters
        """
        if "fuel_level" in config:
            self.fuel_level = config["fuel_level"]

        if "consumption_rates" in config:
            self.consumption_rates.update(config["consumption_rates"])

        if "action_multipliers" in config:
            self.action_multipliers.update(config["action_multipliers"])

        if "craft_status" in config:
            self.craft_status.update(config["craft_status"])

        if "gravity_assist_benefit" in config:
            self.gravity_assist_benefit = config["gravity_assist_benefit"]

        if "orbital_efficiency_factor" in config:
            self.orbital_efficiency_factor = config["orbital_efficiency_factor"]

        if "last_optimisation" in config:
            self.last_optimisation = config["last_optimisation"]

        return True

    def calculate_consumption(self, phase, action, velocity):
        """
        Calculate fuel consumption for the current state and action.

        Args:
            phase: Current mission phase
            action: Action being performed
            velocity: Current velocity vector

        Returns:
            Amount of fuel consumed
        """
        # Base consumption rate for the phase
        base_rate = self.consumption_rates.get(phase, 0.02)

        # Multiply by the action-specific factor
        action_multiplier = self.action_multipliers.get(action, 1.0)

        # Factor in velocity (higher speed = higher consumption)
        speed = np.linalg.norm(velocity)
        velocity_factor = 1.0 + (speed * 0.1)  # 10% increase per unit of speed

        consumption = base_rate * action_multiplier * velocity_factor
        return consumption

    def calculate_gravitational_force(self, position):
        """
        Calculate the net gravitational force on the spacecraft from all celestial bodies.

        """

        net_force = np.zeros(3)

        for body_id, body_data in self.astro_data.celestial_objects.items():
            body_position = np.array(body_data.get("position", [0, 0, 0]))
            body_mass = body_data.get("mass", 0)

            r_vector = body_position - position
            r_magnitude = np.linalg.norm(r_vector)

            if r_magnitude > 0.0:
                force_magnitude = G * body_mass * self.craft_status["mass"] / (r_magnitude ** 2)
                force_direction = r_vector / r_magnitude
                net_force += force_magnitude * force_direction

        return net_force

    def is_in_gravity_assist_zone(self, body_id, position):
        """Determine if the spacecraft is in a gravity assist zone of a celestial body."""
        body_data = self.astro_data.get_body_data(body_id)
        if not body_data:
            return False

        body_position = np.array(body_data.get("position", [0, 0, 0]))
        body_mass = body_data.get("mass", 0)

        distance = np.linalg.norm(position - body_position)

        # Define gravity assist zone based on the body's mass
        influence_zone = np.cbrt(body_mass / 1e23) * 1e6  # Simple formula for influence zone

        return distance < influence_zone * 3 and distance > influence_zone * 0.8

    def calculate_orbital_parameters(self, position, velocity, central_body_id):
        """Calculate orbital parameters (semi-major axis, eccentricity) around a central body."""
        body_data = self.astro_data.get_body_data(central_body_id)
        if not body_data:
            return None

        body_position = np.array(body_data.get("position", [0, 0, 0]))
        body_mass = body_data.get("mass", 0)

        r_v = position - body_position
        r = np.linalg.norm(r_v)
        v = np.linalg.norm(velocity)

        mu = G * body_mass
        specific_energy = (v ** 2) / 2 - mu / r

        # Semi-major axis
        if specific_energy != 0:
            a = -mu / (2 * specific_energy)
        else:
            a = float("inf")  # parabolic orbit

        # Angular momentum
        h_v = np.cross(r_v, velocity)
        h = np.linalg.norm(h_v)

        # Eccentricity
        e_v = np.cross(velocity, h_v) / mu - r_v / r
        e = np.linalg.norm(e_v)

        return {
            "semi_major_axis": a,
            "eccentricity": e,
            "specific_energy": specific_energy,
            "angular_momentum": h
        }

    @staticmethod
    def hohmann_transfer_delta_v(r1, r2, mu):
        """Calculate the delta-v required for a Hohmann transfer between two circular orbits."""

        v1 = np.sqrt(mu / r1)
        v2 = np.sqrt(mu / r2)

        a_transfer = (r1 + r2) / 2

        delta_v1 = np.sqrt(mu / r1) * (np.sqrt(2 * r2 / (r1 + r2)) - 1)
        delta_v2 = np.sqrt(mu / r2) * (1 - np.sqrt(2 * r1 / (r1 + r2)))

        total_delta_v = abs(delta_v1) + abs(delta_v2)
        return total_delta_v

    @staticmethod
    def bi_elliptic_transfer_delta_v(r1, r2, rb, mu):
        """Calculate delta-v for a bi-elliptic transfer (more efficient for large orbit changes)."""
        v1 = np.sqrt(mu / r1) * (np.sqrt(2 * rb / (r1 + rb)) - 1)
        v2 = np.sqrt(mu / rb) * (np.sqrt(2 * rb/ (r2 + rb)) - 1)
        v3 = np.sqrt(mu / r2) * (1 -np.sqrt(2 * rb / (r2 + rb)))

        total_delta_v = abs(v1) + abs(v2) + abs(v3)
        return total_delta_v



    def optimise_trajectory(self, current_position, target_position, current_fuel, phase):
        """
        Optimise the trajectory based on current and target positions, fuel level, and mission phase.

        Args:
            current_position: Current position of the spacecraft
            target_position: Target position for the spacecraft
            current_fuel: Current fuel level
            phase: Current mission phase

        Returns:
            New trajectory vector
        """

        current_position = np.array(current_position)
        target_position = np.array(target_position)
        distance = np.linalg.norm(target_position - current_position)

        # Find the nearest dominant celestial body
        nearest_body = None
        nearest_distance = float("inf")

        for body_id, body_data in self.astro_data.celestial_objects.items():
            body_position = np.array(body_data.get("position", [0, 0, 0]))
            body_distance = np.linalg.norm(current_position - body_position)

            if body_distance < nearest_distance:
                nearest_distance = body_distance
                nearest_body = body_id

        # Check if any gravity assists are possible
        possible_assists = []
        for body_id in self.astro_data.celestial_objects:
            if body_id != nearest_body and self.is_in_gravity_assist_zone(body_id, current_position):
                possible_assists.append(body_id)

        # Calculate optimal speed based on distance and phase
        if phase == "EMERGENCY" or current_fuel < 0.2:
            optimal_speed = 0.5
            recommended_action = "maintain_course"
        elif phase == "LAUNCH":
            optimal_speed = 1.5
            recommended_action = "increase_velocity"
        elif phase == "TRANSIT" and distance > 10.0:
            # For transit, consider orbital mechanics
            if nearest_body and nearest_distance < 5e6:  # If close to a celestial body
                # Use the body's gravity to optimise trajectory
                if possible_assists:
                    # Gravity assist available
                    recommended_action = "adjust_trajectory"
                    optimal_speed = 1.2  # Higher speed for gravity assist
                else:
                    # Use orbital mechanics for efficiency
                    orbital_params = self.calculate_orbital_parameters(
                        current_position, self.craft_status["velocity"], nearest_body
                    )
                    if orbital_params and orbital_params["eccentricity"] > 0.1:
                        recommended_action = "maintain_course"
                        optimal_speed = 0.8  # Lower speed for elliptical orbits
                    else:
                        recommended_action = "adjust_trajectory"
                        optimal_speed = 1.0  # Default speed for transit
            else:
                # No nearby body, maintain course
                recommended_action = "maintain_course"
                optimal_speed = 1.0
        elif phase == "EXPLORATION":
            optimal_speed = 0.7
            recommended_action = "decrease_velocity"
        else:
            optimal_speed = 1.0
            recommended_action = "maintain_course"

        # Calculate estimated fuel usage for the journey
        efficiency_multiplier = 1.0
        if nearest_body:
            body_data = self.astro_data.get_body_data(nearest_body)
            if body_data and "mass" in body_data:
                mu = G * body_data["mass"]
                r1 = nearest_distance

                nearest_body_position = np.array(body_data.get("position", [0, 0, 0]))
                r2 = np.linalg.norm(target_position - nearest_body_position)

                # Compare direct transfer to orbital transfer
                direct_consumption = distance * self.consumption_rates.get(phase, 0.02)

                # If the spacecraft is in proper orbit conditions, use that to our advantage
                if r1 > 2e6 and r2 > 1e6:  # Avoid very close approaches
                    hohmann_transfer = self.hohmann_transfer_delta_v(r1, r2, mu)
                    orbital_consumption = hohmann_transfer * 0.02  # Assume 2% fuel per delta-v unit

                    if orbital_consumption < direct_consumption:
                        efficiency_multiplier = orbital_consumption / direct_consumption
                        recommended_action = "adjust_trajectory"

                # Check if gravity assist is possible and factor in the benefit
                if possible_assists:
                    efficiency_multiplier *= (1 - self.gravity_assist_benefit)  # Reduce consumption
                    recommended_action = "adjust_trajectory"

        estimated_consumption = (distance / optimal_speed) * self.consumption_rates.get(phase,
                                                                                        0.02) * efficiency_multiplier

        # Store this optimisation for reference
        self.last_optimisation = {
            "phase": phase,
            "distance": distance,
            "optimal_speed": optimal_speed,
            "estimated_consumption": estimated_consumption,
            "recommended_action": recommended_action,
            "nearest_body": nearest_body,
            "nearest_distance": nearest_distance,
            "gravity_assists": possible_assists,
            "efficiency_multiplier": efficiency_multiplier
        }

        return self.last_optimisation

    def refuel(self, amount):
        """Add fuel from the refuelling operation."""
        self.fuel_level = min(1.0, self.fuel_level + amount)
        return self.fuel_level

    def update_consumption_model(self, phase_data):
        """
        Update consumption rates based on observed data.

        Args:
            phase_data: Dictionary with phase consumption statistics
        """
        for phase, rate in phase_data.items():
            if phase in self.consumption_rates:
                # Update the model with 20% weight to new data
                self.consumption_rates[phase] = (
                        self.consumption_rates[phase] * 0.8 + rate * 0.2
                )

        return self.consumption_rates


    def predict_orbital_path(self, current_position, velocity, central_body_id, time_steps=100, step_size=3600):
        """Predict orbital path over time using numerical integration."""
        body_data = self.astro_data.get_body_data(central_body_id)
        if not body_data or "position" not in body_data or "mass" not in body_data:
            logging.warning(f"Cannot predict orbital path: incomplete data for {central_body_id}")
            return []

        body_position = np.array(body_data.get("position"))
        body_mass = body_data.get("mass")

        orbital_path = []
        position = np.array(current_position)
        velocity = np.array(velocity)

        for _ in range(time_steps):
            orbital_path.append(position.copy())

            r_v = position - body_position
            r = np.linalg.norm(r_v)

            if r > 0:
                mu = G * body_mass
                acc = -mu * r_v / r ** 3

                velocity += acc * step_size
                position += velocity * step_size

        return orbital_path



    def get_efficiency_metrics(self):
        """Return current efficiency metrics."""
        if not self.efficiency_history:
            return {"avg_consumption": 0, "efficiency_trend": "stable"}

        recent = self.efficiency_history[-10:]
        if len(recent) >= 2:
            trend = "improving" if recent[-1] < recent[0] else "worsening"
        else:
            trend = "stable"

        return {
            "avg_consumption": sum(recent) / len(recent),
            "efficiency_trend": trend,
            "current_model": {p: round(r, 4) for p, r in self.consumption_rates.items()}
        }

