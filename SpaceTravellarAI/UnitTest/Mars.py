import numpy as np
import logging
from teg import run_mars_mission


def configure_mars_mission():
    """
    Configure and execute a Mars exploration mission with specific objectives
    and waypoints for the AI Captain to navigate.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='mars_mission.log'
    )
    logger = logging.getLogger('mars_mission')
    logger.info("Configuring Mars mission parameters")

    # Define mission waypoints - distances in millions of kilometers
    # Mars is approximately 225 million km from Earth at its closest
    waypoints = [
        {
            "name": "Earth Orbit",
            "position": np.array([0.0, 0.0, 0.0]),
            "description": "Starting point in Earth orbit"
        },
        {
            "name": "Lunar Flyby",
            "position": np.array([0.4e9, 0.0, 0.0]),
            "description": "Gravity assist maneuver around the Moon"
        },
        {
            "name": "Deep Space Checkpoint",
            "position": np.array([80e9, 10e9, 0.0]),
            "description": "Deep space navigation checkpoint"
        },
        {
            "name": "Mars Approach",
            "position": np.array([220e9, 20e9, 5e9]),
            "description": "Initial approach to Mars orbit"
        },
        {
            "name": "Mars Orbit",
            "position": np.array([225e9, 25e9, 5e9]),
            "description": "Stable orbit around Mars"
        },
        {
            "name": "Phobos Rendezvous",
            "position": np.array([225.2e9, 25.2e9, 5.1e9]),
            "description": "Rendezvous with Mars' moon Phobos"
        },
        {
            "name": "Mars Landing Site",
            "position": np.array([225.5e9, 25.5e9, 5.0e9]),
            "description": "Final landing site in Jezero Crater"
        }
    ]

    # Mission parameters
    mission_params = {
        "mission_name": "Mars Exploration Alpha",
        "mission_objective": "Navigate to Mars, survey Phobos, and land at Jezero Crater",
        "mission_priority": "Fuel efficiency and scientific exploration",
        "steps": 500,  # Longer mission for complex trajectory
        "initial_fuel": 1.0,
        "anomaly_probability": 0.25,  # Higher likelihood of anomalies for a challenging mission
        "waypoints": waypoints
    }

    logger.info(f"Mission configured with {len(waypoints)} waypoints")
    logger.info(f"Primary objective: {mission_params['mission_objective']}")

    # Print mission briefing
    print("\n======== MARS MISSION BRIEFING ========")
    print(f"Mission: {mission_params['mission_name']}")
    print(f"Objective: {mission_params['mission_objective']}")
    print(f"Priority: {mission_params['mission_priority']}")
    print(f"Waypoints: {len(waypoints)}")
    for i, wp in enumerate(waypoints):
        print(f"  {i + 1}. {wp['name']} - {wp['description']}")
    print("=======================================\n")

    # Run the mission with our parameters
    # Pass only the 'steps' parameter, as 'waypoints' isn't accepted
    captain = run_mars_mission(
        steps=mission_params["steps"]
    )

    # Report mission results
    print("\n======== MISSION REPORT ========")
    print(f"Final position: {captain.sensor_data.current_readings['position']}")
    print(f"Remaining fuel: {captain.fuel_optimiser.fuel_level:.2f}")
    print(f"Final phase: {captain.current_phase}")
    print("===============================\n")

    return captain

if __name__ == "__main__":
    captain = configure_mars_mission()