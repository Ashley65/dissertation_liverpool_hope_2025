import numpy as np
import matplotlib.pyplot as plt
from optimiser import FuelOptimiser


class AstroDataManager:
    """Mock astronomical data manager for testing purposes."""

    def __init__(self):
        # Create a solar system with Sun, Earth, Mars and Jupiter
        self.celestial_objects = {
            "Sun": {
                "position": [0, 0, 0],
                "mass": 1.989e30,  # kg
                "radius": 696340e3  # meters
            },
            "Earth": {
                "position": [1.496e11, 0, 0],  # 1 AU from Sun
                "mass": 5.972e24,
                "radius": 6371e3
            },
            "Mars": {
                "position": [2.279e11, 0, 0],  # 1.52 AU from Sun
                "mass": 6.39e23,
                "radius": 3389.5e3
            },
            "Jupiter": {
                "position": [7.785e11, 0, 0],  # 5.2 AU from Sun
                "mass": 1.898e27,
                "radius": 69911e3
            }
        }

    def get_body_data(self, body_id):
        """Get data for a specific celestial body."""
        return self.celestial_objects.get(body_id)


def run_mission_simulation():
    """Run a simulated mission to test the FuelOptimiser."""
    print("Starting Mission Simulation Test...")

    # Initialize the astronomical data and fuel optimizer
    astro_data = AstroDataManager()
    optimiser = FuelOptimiser(astro_data, initial_fuel_level=1.0)

    # Mission parameters
    mission_phases = ["LAUNCH", "TRANSIT", "EXPLORATION", "TRANSIT", "CRITICAL", "EMERGENCY"]
    mission_targets = [
        np.array([1.496e10, 0, 0]),  # Initial target in Earth orbit
        np.array([2.279e11, 0, 0]),  # Mars position
        np.array([2.279e11, 1e7, 0]),  # Mars orbit exploration
        np.array([7.785e11, 0, 0]),  # Jupiter position
        np.array([1.496e11, 0, 0]),  # Return to Earth
        np.array([1.496e10, 0, 0])  # Final Earth orbit
    ]

    # Store trajectory and fuel data for analysis
    trajectory = []
    fuel_levels = []
    consumption_rates = []

    # Initialize spacecraft position at Earth orbit
    current_position = np.array([1.496e10, 0, 0])  # Near Earth

    # Run through mission phases
    for i, (phase, target) in enumerate(zip(mission_phases, mission_targets)):
        print(f"\n--- Mission Phase: {phase} ---")
        print(f"Current position: {current_position}")
        print(f"Target position: {target}")
        print(f"Current fuel level: {optimiser.fuel_level:.2f}")

        # Get optimization plan
        optimization = optimiser.optimise_trajectory(
            current_position,
            target,
            optimiser.fuel_level,
            phase
        )

        # Print optimization details
        print("\nOptimization Plan:")
        for key, value in optimization.items():
            print(f"  {key}: {value}")

        # Simulate movement toward target
        distance = np.linalg.norm(target - current_position)
        steps = 10  # Simulate in 10 steps

        for step in range(steps):
            # Move position toward target
            direction = (target - current_position) / distance
            step_distance = distance / steps
            current_position = current_position + direction * step_distance

            # Calculate and consume fuel for this step
            action = optimization["recommended_action"]
            velocity = direction * optimization["optimal_speed"]
            consumption = optimiser.calculate_consumption(phase, action, velocity)
            optimiser.fuel_level -= consumption / steps

            # Store data for plotting
            trajectory.append(current_position.copy())
            fuel_levels.append(optimiser.fuel_level)
            consumption_rates.append(consumption)

            # Check for emergency fuel conditions
            if optimiser.fuel_level < 0.1 and phase != "EMERGENCY":
                print("\n!!! EMERGENCY FUEL LEVELS DETECTED - SWITCHING TO EMERGENCY PROTOCOL !!!")
                phase = "EMERGENCY"
                # Re-optimize with emergency protocols
                optimization = optimiser.optimise_trajectory(
                    current_position,
                    target,
                    optimiser.fuel_level,
                    phase
                )

        # Check if we need to simulate refueling at certain destinations
        if i == 2:  # After Mars exploration
            print("\nRefueling operation at Mars base")
            optimiser.refuel(0.4)
            print(f"New fuel level: {optimiser.fuel_level:.2f}")

        # Update the efficiency history
        if hasattr(optimiser, 'efficiency_history'):
            optimiser.efficiency_history.append(consumption)

    # Show final results
    print("\n--- Mission Complete ---")
    print(f"Final position: {current_position}")
    print(f"Final fuel level: {optimiser.fuel_level:.2f}")

    efficiency_metrics = optimiser.get_efficiency_metrics()
    print("\nEfficiency Metrics:")
    for key, value in efficiency_metrics.items():
        print(f"  {key}: {value}")

    # Plot the results
    plot_mission_results(trajectory, fuel_levels, consumption_rates)


def plot_mission_results(trajectory, fuel_levels, consumption_rates):
    """Plot the mission trajectory and fuel consumption."""
    fig = plt.figure(figsize=(15, 10))

    # 3D plot of trajectory
    ax1 = fig.add_subplot(221, projection='3d')
    trajectory = np.array(trajectory)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', label='Spacecraft Path')
    ax1.set_title('Mission Trajectory')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()

    # 2D plot of trajectory (X-Y plane)
    ax2 = fig.add_subplot(222)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'g-')
    ax2.set_title('Trajectory (X-Y Plane)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')

    # Plot fuel levels over time
    ax3 = fig.add_subplot(223)
    ax3.plot(fuel_levels, 'r-')
    ax3.set_title('Fuel Level Over Time')
    ax3.set_xlabel('Mission Step')
    ax3.set_ylabel('Fuel Level')
    ax3.set_ylim(0, 1.1)

    # Plot consumption rates over time
    ax4 = fig.add_subplot(224)
    ax4.plot(consumption_rates, 'b-')
    ax4.set_title('Fuel Consumption Rate')
    ax4.set_xlabel('Mission Step')
    ax4.set_ylabel('Consumption Rate')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_mission_simulation()