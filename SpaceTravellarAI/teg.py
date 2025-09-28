import numpy as np
import matplotlib.pyplot as plt
from main import AICaptain
from astroDataManager import AstroDataManager
from optimiser import FuelOptimiser

import logging
import traceback
import inspect

from sensor import SensorData


# 1. Set up a proper logging system
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("mars_mission.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("MarsMission")


# 2. Add a state inspector function
def inspect_state(captain, current_waypoint, step):
    """Print detailed state information for debugging"""
    logger = logging.getLogger("MarsMission")

    logger.debug(f"--- STATE INSPECTION (Step {step}) ---")
    logger.debug(f"Current position: {captain.sensor_data.current_readings['position']}")
    if current_waypoint < len(captain.sensor_data.current_readings['mission_waypoints']):
        target = captain.sensor_data.current_readings['mission_waypoints'][current_waypoint]['position']
        logger.debug(f"Target position: {target}")
        # Validate position data types
        logger.debug(f"Position type: {type(captain.sensor_data.current_readings['position'])}")
        logger.debug(f"Target position type: {type(target)}")
    logger.debug(f"Current phase: {captain.current_phase}")
    logger.debug(f"Fuel level: {captain.fuel_optimiser.fuel_level}")


def inspect_state(captain, current_waypoint, step):
    """Inspect the captain's state at the current step for debugging."""
    logger = logging.getLogger("MarsMission")
    logger.info(f"=== Step {step} - Inspection ===")
    logger.info(f"Current phase: {captain.current_phase}")

    # Safely access waypoints
    waypoints = captain.sensor_data.current_readings.get('mission_waypoints', [])
    if current_waypoint < len(waypoints):
        logger.info(f"Current waypoint: {waypoints[current_waypoint]['name']}")
    else:
        logger.info(f"Current waypoint index: {current_waypoint} (out of range)")

    # Log current position and fuel
    position = captain.sensor_data.current_readings.get('position', np.array([0, 0]))
    fuel = captain.sensor_data.current_readings.get('fuel_level', 0)
    logger.info(f"Position: {position}")
    logger.info(f"Fuel level: {fuel:.2f}")

    # Additional debug information
    logger.info(f"Sensor data keys: {list(captain.sensor_data.current_readings.keys())}")
    logger.info("=== End of Inspection ===")


def validate_numpy_array(arr, name):
    """Ensure the given value is a proper numpy array."""
    if arr is None:
        raise ValueError(f"{name} is None")

    if not isinstance(arr, np.ndarray):
        try:
            # Try to convert to numpy array if it's not already
            return np.array(arr)
        except:
            raise TypeError(f"{name} is not a numpy array and cannot be converted to one")

    return arr


def setup_mars_mission(captain):
    """Set up a Mars mission with the AI Captain."""
    # Initialize AstroDataManager to get positions
    astro_manager = AstroDataManager()

    # Check if astro_manager has the required data
    if not captain.check_astro_data_manager():
        # Fall back to approximate positions if celestial data is missing
        earth_position = np.array([1.0, 0.0, 0.0]) * 1.5e11  # Ensure it's a numpy array
        mars_position = np.array([1.5, 0.0, 0.0]) * 2.2e11  # Ensure it's a numpy array
        logging.warning("Using fallback positions for Earth and Mars")
    else:
        try:
            earth_data = astro_manager.get_body_data("Earth")
            mars_data = astro_manager.get_body_data("Mars")

            # Convert positions to numpy arrays if they aren't already
            earth_position = np.array(earth_data["position"])
            mars_position = np.array(mars_data["position"])
        except (KeyError, TypeError) as e:
            # Fall back to approximate positions if data retrieval fails
            earth_position = np.array([1.0, 0.0, 0.0]) * 1.5e11
            mars_position = np.array([1.5, 0.0, 0.0]) * 2.2e11
            logging.warning(f"Error retrieving planet data: {e}. Using fallback positions.")

    # Define mission waypoints with proper numpy array operations
    waypoints = [
        {'position': np.array(earth_position) * 0.1, 'name': 'Earth Orbit', 'phase': 'LAUNCH'},
        {'position': np.array(earth_position) * 0.5 + np.array(mars_position) * 0.5, 'name': 'Transit Midpoint',
         'phase': 'TRANSIT'},
        {'position': np.array(mars_position), 'name': 'Mars Arrival', 'phase': 'CRITICAL'},
        {'position': np.array(mars_position) + np.array([1e7, 0, 0]), 'name': 'Mars Exploration Site A',
         'phase': 'EXPLORATION'},
        {'position': np.array(mars_position) * 0.5 + np.array(earth_position) * 0.5, 'name': 'Return Transit',
         'phase': 'CRITICAL'},
        {'position': np.array(earth_position), 'name': 'Earth Return', 'phase': 'CRITICAL'},
    ]

    # Initialize sensor data if not already set
    if not hasattr(captain, 'sensor_data') or not hasattr(captain.sensor_data, 'current_readings'):
        captain.sensor_data = SensorData({})
        captain.sensor_data.current_readings = {}

    # Add required keys to the sensor data
    captain.sensor_data.current_readings['mission_waypoints'] = waypoints
    captain.sensor_data.current_readings['target_position'] = waypoints[0]['position']
    captain.sensor_data.current_readings['position'] = np.array(earth_position) * 0.01  # Starting near Earth
    captain.sensor_data.current_readings['fuel_level'] = 1.0  # Full tank
    captain.sensor_data.current_readings['velocity'] = np.array([0.0, 0.0, 0.0])  # Starting at rest

    # Set initial mission phase
    captain.current_phase = waypoints[0]['phase']
    captain.current_waypoint = 0

    logging.info("Mars mission setup complete")
    return waypoints


def initialize_spacecraft_data(data_manager):
    """Initialize spacecraft data and add it to the celestial objects"""
    # Create basic spacecraft data
    spacecraft_data = {
        "position": np.array([1.5e11, 0, 0]).tolist(),  # Start at origin or Earth's position
        "velocity": np.array([0, 0, 0]).tolist(),
        "mass": 1000.0,  # kg
        "fuel": 100.0,  # percentage
        "status": "READY"
    }

    # Add spacecraft to the celestial objects database
    data_manager.add_body_data("spacecraft", spacecraft_data)

    return spacecraft_data


def run_mars_mission(steps=100, debug=False):
    """
    Run a Mars exploration mission with the AI Captain.
    """
    # Set up logging
    logger = logging.getLogger("MarsMission")
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.info("Starting Mars Mission Simulation...")

    # Initialize the captain
    captain = AICaptain()

    # Set up the mission and get waypoints
    try:
        waypoints = setup_mars_mission(captain)
        current_waypoint = 0

        # Ensure waypoints are accessible in the captain's sensor data
        if 'mission_waypoints' not in captain.sensor_data.current_readings:
            captain.sensor_data.current_readings['mission_waypoints'] = waypoints

        # Ensure target_position is set
        if 'target_position' not in captain.sensor_data.current_readings:
            captain.sensor_data.current_readings['target_position'] = waypoints[0]['position']

        trajectory = []
        fuel_levels = []
        actions_taken = []
        phases = []
        completion_threshold = 1e9  # 1 million km

        for step in range(steps):
            # Debug state at each step if in debug mode
            if debug and step % 10 == 0:
                inspect_state(captain, current_waypoint, step)

            try:
                # Ensure position is a numpy array
                position = captain.sensor_data.current_readings.get("position", np.array([0, 0]))
                trajectory.append(position.copy())
                fuel_levels.append(captain.sensor_data.current_readings.get("fuel_level", 0))
                phases.append(captain.current_phase)

                # Validate target position exists
                if "target_position" not in captain.sensor_data.current_readings:
                    captain.sensor_data.current_readings["target_position"] = waypoints[current_waypoint]['position']

                # Validate and get target position
                target_pos = validate_numpy_array(
                    captain.sensor_data.current_readings["target_position"],
                    "target_position"
                )

                # Calculate distance with proper error handling
                try:
                    distance = np.linalg.norm(position - target_pos)
                    logger.info(f"Distance to waypoint: {distance / 1e9:.2f} million km")
                except Exception as e:
                    logger.error(f"Error calculating distance: {e}")
                    logger.error(f"Position: {position}, type: {type(position)}")
                    logger.error(f"Target position: {target_pos}, type: {type(target_pos)}")
                    raise

                # Check if we've reached the current waypoint
                if distance < completion_threshold:
                    current_waypoint += 1
                    if current_waypoint < len(waypoints):
                        waypoint = waypoints[current_waypoint]
                        logger.info(f"Waypoint reached! Moving to {waypoint['name']} (Phase: {waypoint['phase']})")
                        captain.current_waypoint = current_waypoint
                        captain.current_phase = waypoint['phase']
                        captain.sensor_data.current_readings["target_position"] = waypoint['position']

                        # Special handling for Mars base
                        if waypoint['name'] == "Mars Exploration Site A":
                            logger.info("At Mars base. Refueling...")
                            # Refuel if you have a fuel optimizer
                            if hasattr(captain, 'fuel_optimiser'):
                                captain.fuel_optimiser.refuel(0.4)
                                captain.sensor_data.current_readings["fuel_level"] = captain.fuel_optimiser.fuel_level
                            else:
                                # Direct refueling if no fuel optimizer
                                captain.sensor_data.current_readings["fuel_level"] = 1.0
                    else:
                        logger.info("Mission Complete! Successfully returned to Earth!")
                        break

                # Run mission step
                action, reward = captain.take_action()
                actions_taken.append(action)
                logger.info(f"Action taken: {action}, reward: {reward}")

                # Update position based on action (simplified physics)
                current_pos = captain.sensor_data.current_readings.get("position", np.array([0, 0]))
                velocity = captain.sensor_data.current_readings.get("velocity", np.array([0, 0]))

                # Update position based on velocity (simplified)
                new_pos = current_pos + velocity
                captain.sensor_data.current_readings["position"] = new_pos

                # Consume fuel based on action
                fuel_level = captain.sensor_data.current_readings.get("fuel_level", 1.0)
                captain.sensor_data.current_readings["fuel_level"] = max(0, fuel_level - 0.01)

                # Inject random anomalies occasionally
                if step % 50 == 0 and step > 0:
                    random_value = np.random.uniform(0, 1)
                    anomaly_detected = random_value < 0.3
                    if anomaly_detected:
                        logger.warning("Anomaly detected in spacecraft systems!")
                        captain.sensor_data.current_readings["anomaly_score"] = float(np.random.uniform(0.6, 0.9))

            except Exception as e:
                logger.error(f"Mission error: {str(e)}")
                logger.error(f"Error occurred at step {step}")
                import traceback
                logger.error(f"Stack trace: {traceback.format_exc()}")
                raise

        return captain

    except Exception as e:
        logger.error(f"Mission setup error: {str(e)}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise


def plot_mission_results(trajectory, fuel_levels, actions_taken, phases, waypoints):
    """Plot the mission trajectory and fuel consumption."""
    trajectory = np.array(trajectory)

    fig = plt.figure(figsize=(15, 10))

    # 2D plot of trajectory
    ax1 = fig.add_subplot(221)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Spacecraft Path')

    # Add waypoints to the plot
    waypoint_positions = np.array([wp["position"] for wp in waypoints])
    ax1.scatter(waypoint_positions[:, 0], waypoint_positions[:, 1], c='red', marker='*', s=100, label='Waypoints')

    # Label waypoints
    for i, wp in enumerate(waypoints):
        ax1.annotate(wp["name"], (wp["position"][0], wp["position"][1]),
                     textcoords="offset points", xytext=(0, 10), ha='center')

    ax1.set_title('Mission Trajectory (X-Y Plane)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend()

    # Plot fuel levels over time
    ax2 = fig.add_subplot(222)
    ax2.plot(fuel_levels, 'r-')
    ax2.set_title('Fuel Level Over Time')
    ax2.set_xlabel('Mission Step')
    ax2.set_ylabel('Fuel Level')
    ax2.set_ylim(0, 1.1)

    # Plot actions taken
    ax3 = fig.add_subplot(223)
    unique_actions = list(set(actions_taken))
    action_counts = [actions_taken.count(action) for action in unique_actions]
    ax3.bar(range(len(unique_actions)), action_counts, tick_label=unique_actions)
    ax3.set_title('Actions Taken During Mission')
    ax3.set_xlabel('Action')
    ax3.set_ylabel('Count')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Plot mission phases
    ax4 = fig.add_subplot(224)
    unique_phases = list(set(phases))
    phase_segments = []
    current_phase = phases[0]
    segment_start = 0

    for i, phase in enumerate(phases):
        if phase != current_phase or i == len(phases) - 1:
            phase_segments.append((segment_start, i, current_phase))
            segment_start = i
            current_phase = phase

    for start, end, phase in phase_segments:
        ax4.axvspan(start, end, alpha=0.3, label=phase if phase not in [p for _, _, p in phase_segments[
                                                                                         :phase_segments.index(
                                                                                             (start, end,
                                                                                              phase))]] else "")

    ax4.set_title('Mission Phases')
    ax4.set_xlabel('Mission Step')
    ax4.set_ylabel('Phase')
    ax4.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    captain = run_mars_mission(steps=300)