#ifndef SIMULATION_MANAGER_CUH
#define SIMULATION_MANAGER_CUH
#pragma once

#include "../physics/Body.cuh"
#include "../physics/NetworkingClient.h"
#include "../ai/AIDataPacket.cuh"
#include "../physics/spaceship.cuh"
#include "physics/physics_engine.cuh"

class SimulationManager {
private:
    physics_engine engine;
    std::vector<Body> bodies;
    std::vector<spaceship> ships;
    firmwareClient aiClient;
    CaptainDataPacket captainData;

    // Environmental parameters
    float temperature = 0.0f;
    float radiation = 0.0f;
    float magneticField = 0.0f;
    std::vector<std::string> hazards;

    // Mission tracking
    MissionPhase currentPhase = MissionPhase::LAUNCH;
    float missionProgress = 0.0f;
    std::vector<Waypoint> missionWaypoints;

    // Anomaly tracking
    bool anomalyPresent = false;
    std::string anomalyType = "";
    float anomalySeverity = 0.0f;
    float anomalyConfidence = 0.0f;

    // Performance metrics
    float rewardAccumulated = 0.0f;
    float pathEfficiency = 0.0f;
    float fuelEfficiency = 0.0f;
    float anomalyHandlingScore = 0.0f;

    // Memory records
    std::vector<MemoryRecord> memoryRecords;

    void initializeSolarSystem() {
        // Clear any existing bodies
        bodies.clear();

        // Create the Sun
        Body sun;
        sun.x = 0.0f; sun.y = 0.0f; sun.z = 0.0f;
        sun.vx = 0.0f; sun.vy = 0.0f; sun.vz = 0.0f;
        sun.mass = 1.989e30f;  // Solar mass
        sun.radius = 6.957e8f;  // Solar radius
        bodies.push_back(sun);

        // Mercury
        Body mercury;
        mercury.x = 5.791e10f; mercury.y = 0.0f; mercury.z = 0.0f;
        mercury.vx = 0.0f; mercury.vy = 4.7362e4f; mercury.vz = 0.0f;
        mercury.mass = 3.3011e23f;
        mercury.radius = 2.4397e6f;
        bodies.push_back(mercury);

        // Venus
        Body venus;
        venus.x = 1.0821e11f; venus.y = 0.0f; venus.z = 0.0f;
        venus.vx = 0.0f; venus.vy = 3.5022e4f; venus.vz = 0.0f;
        venus.mass = 4.8675e24f;
        venus.radius = 6.0518e6f;
        bodies.push_back(venus);

        // Earth
        Body earth;
        earth.x = 1.496e11f; earth.y = 0.0f; earth.z = 0.0f;
        earth.vx = 0.0f; earth.vy = 2.9783e4f; earth.vz = 0.0f;
        earth.mass = 5.972e24f;
        earth.radius = 6.371e6f;
        bodies.push_back(earth);

        // Mars
        Body mars;
        mars.x = 2.2794e11f; mars.y = 0.0f; mars.z = 0.0f;
        mars.vx = 0.0f; mars.vy = 2.4077e4f; mars.vz = 0.0f;
        mars.mass = 6.4171e23f;
        mars.radius = 3.3895e6f;
        bodies.push_back(mars);

        // Jupiter
        Body jupiter;
        jupiter.x = 7.7857e11f; jupiter.y = 0.0f; jupiter.z = 0.0f;
        jupiter.vx = 0.0f; jupiter.vy = 1.3072e4f; jupiter.vz = 0.0f;
        jupiter.mass = 1.8982e27f;
        jupiter.radius = 6.9911e7f;
        bodies.push_back(jupiter);

        // Saturn
        Body saturn;
        saturn.x = 1.4335e12f; saturn.y = 0.0f; saturn.z = 0.0f;
        saturn.vx = 0.0f; saturn.vy = 9.6528e3f; saturn.vz = 0.0f;
        saturn.mass = 5.6834e26f;
        saturn.radius = 5.8232e7f;
        bodies.push_back(saturn);

        // Uranus
        Body uranus;
        uranus.x = 2.8725e12f; uranus.y = 0.0f; uranus.z = 0.0f;
        uranus.vx = 0.0f; uranus.vy = 6.8352e3f; uranus.vz = 0.0f;
        uranus.mass = 8.6810e25f;
        uranus.radius = 2.5362e7f;
        bodies.push_back(uranus);

        // Neptune
        Body neptune;
        neptune.x = 4.4951e12f; neptune.y = 0.0f; neptune.z = 0.0f;
        neptune.vx = 0.0f; neptune.vy = 5.4778e3f; neptune.vz = 0.0f;
        neptune.mass = 1.0243e26f;
        neptune.radius = 2.4622e7f;
        bodies.push_back(neptune);
    }

    // Initialize mission waypoints
    void initializeWaypoints() {
        missionWaypoints.clear();

        // Define waypoints for Mars mission
        Waypoint earth_orbit;
        earth_orbit.x = 1.496e11f;
        earth_orbit.y = 1.0e10f;
        earth_orbit.z = 0.0f;
        earth_orbit.name = "Earth Orbit";
        earth_orbit.reached = false;
        missionWaypoints.push_back(earth_orbit);

        Waypoint transfer_midpoint;
        transfer_midpoint.x = 1.9e11f;
        transfer_midpoint.y = 0.0f;
        transfer_midpoint.z = 0.0f;
        transfer_midpoint.name = "Transfer Midpoint";
        transfer_midpoint.reached = false;
        missionWaypoints.push_back(transfer_midpoint);

        Waypoint mars_approach;
        mars_approach.x = 2.2794e11f;
        mars_approach.y = -1.0e10f;
        mars_approach.z = 0.0f;
        mars_approach.name = "Mars Approach";
        mars_approach.reached = false;
        missionWaypoints.push_back(mars_approach);

        Waypoint mars_orbit;
        mars_orbit.x = 2.2794e11f;
        mars_orbit.y = 0.0f;
        mars_orbit.z = 1.0e10f;
        mars_orbit.name = "Mars Orbit";
        mars_orbit.reached = false;
        missionWaypoints.push_back(mars_orbit);
    }

    // Update the captain data packet with current simulation state
    void updateCaptainData(double currentTime) {
        // Update from physics data
        captainData.updateFromSimulation(bodies, ships, currentTime);

        // Update mission context
        captainData.updateMissionContext(currentPhase, missionProgress, missionWaypoints);

        // Update environmental data
        captainData.updateEnvironment(temperature, radiation, magneticField, hazards);

        // Update anomaly information
        captainData.updateAnomaly(anomalyPresent, anomalyType, anomalySeverity, anomalyConfidence);

        // Update memory records
        captainData.updateMemories(memoryRecords);

        // Update performance metrics
        captainData.updatePerformance(rewardAccumulated, pathEfficiency,
                                      fuelEfficiency, anomalyHandlingScore);
    }

    // Update environmental conditions based on ship position
    void updateEnvironment() {
        if (ships.empty()) return;

        const spaceship& ship = ships[0];

        // Distance from sun affects temperature and radiation
        float distFromSun = std::sqrt(ship.x*ship.x + ship.y*ship.y + ship.z*ship.z);

        // Temperature decreases with distance from sun
        temperature = 300.0f * (1.5e11f / distFromSun);

        // Radiation increases closer to sun and during solar events
        radiation = 10.0f * (1.5e11f / distFromSun);

        // Magnetic field changes with position
        magneticField = 50.0f / (distFromSun / 1.5e11f);

        // Check for hazards
        hazards.clear();

        // Example hazard detection: asteroid belt
        if (distFromSun > 3.0e11f && distFromSun < 5.0e11f) {
            hazards.push_back("ASTEROID_FIELD");
        }

        // Example solar flare detection
        if (radiation > 20.0f) {
            hazards.push_back("SOLAR_FLARE");
        }

        // Near planet hazards
        for (const auto& body : bodies) {
            float distToBody = std::sqrt(
                std::pow(ship.x - body.x, 2) +
                std::pow(ship.y - body.y, 2) +
                std::pow(ship.z - body.z, 2)
            );

            // If close to a large body
            if (body.mass > 1.0e24f && distToBody < body.radius * 10) {
                hazards.push_back("GRAVITATIONAL_HAZARD");
            }
        }
    }

    // Check for anomalies in the system
    void detectAnomalies() {
        anomalyPresent = false;
        anomalyType = "";
        anomalySeverity = 0.0f;
        anomalyConfidence = 0.0f;

        if (ships.empty()) return;

        const spaceship& ship = ships[0];

        // Example anomaly detection: fuel leak
        if (ship.fuel < 50.0f) {
            anomalyPresent = true;
            anomalyType = "FUEL_LEAK";
            anomalySeverity = (50.0f - ship.fuel) / 50.0f;
            anomalyConfidence = 0.85f;
        }

        // Example anomaly: off-course detection
        if (!missionWaypoints.empty()) {
            const Waypoint& nextWaypoint = missionWaypoints[0];
            float distToWaypoint = std::sqrt(
                std::pow(ship.x - nextWaypoint.x, 2) +
                std::pow(ship.y - nextWaypoint.y, 2) +
                std::pow(ship.z - nextWaypoint.z, 2)
            );

            // If far from expected path
            if (distToWaypoint > 2.0e10f) {
                anomalyPresent = true;
                anomalyType = "TRAJECTORY_DEVIATION";
                anomalySeverity = std::min(distToWaypoint / 5.0e10f, 1.0f);
                anomalyConfidence = 0.9f;
            }
        }
    }

    // Update mission progress and phase
    void updateMissionStatus() {
        if (ships.empty() || missionWaypoints.empty()) return;

        const spaceship& ship = ships[0];

        // Check if current waypoint is reached
        if (!missionWaypoints[0].reached) {
            float distToWaypoint = std::sqrt(
                std::pow(ship.x - missionWaypoints[0].x, 2) +
                std::pow(ship.y - missionWaypoints[0].y, 2) +
                std::pow(ship.z - missionWaypoints[0].z, 2)
            );

            if (distToWaypoint < 1.0e9f) { // 1 million km threshold
                missionWaypoints[0].reached = true;

                // Calculate mission progress based on waypoints reached
                int reachedCount = 0;
                for (const auto& wp : missionWaypoints) {
                    if (wp.reached) reachedCount++;
                }

                missionProgress = static_cast<float>(reachedCount) / missionWaypoints.size();

                // Update mission phase based on progress
                if (missionProgress < 0.25f) {
                    currentPhase = MissionPhase::LAUNCH;
                } else if (missionProgress < 0.75f) {
                    currentPhase = MissionPhase::TRANSIT;
                } else {
                    currentPhase = MissionPhase::EXPLORATION;
                }
            }
        }

        // Check for critical or emergency conditions
        if (anomalyPresent && anomalySeverity > 0.7f) {
            currentPhase = MissionPhase::CRITICAL;
        }

        if (ship.fuel < 10.0f || (anomalyPresent && anomalySeverity > 0.9f)) {
            currentPhase = MissionPhase::EMERGENCY;
        }
    }

public:
    SimulationManager(int maxBodies = 1000, int maxShips = 0,
                     const std::string& aiServerEndpoint = "tcp://localhost:5555")
        : engine(maxBodies, maxShips), aiClient(aiServerEndpoint) {
        // Initialize solar system bodies
        initializeSolarSystem();

        // Initialize mission waypoints
        initializeWaypoints();

        // Initialize a spacecraft
        spaceship ship;
        ship.id = 1;
        ship.x = 1.496e11f; // Start at Earth position
        ship.y = 0.0f;
        ship.z = 0.0f;
        ship.vx = 0.0f;
        ship.vy = 2.9783e4f; // Earth orbital velocity
        ship.vz = 0.0f;
        ship.mass = 1000.0f; // 1000 kg spacecraft
        ship.fuel = 100.0f;  // Full fuel
        ships.push_back(ship);

        // Initialize trajectory tracking (record 100 points at 1-day intervals)
        engine.initTrajectories(bodies.size() + ships.size(), 100, 86400.0f);

        std::cout << "Simulation initialized with " << bodies.size() << " bodies and "
                  << ships.size() << " spacecraft" << std::endl;
    }

    const std::vector<Body>& getBodies() const {
        return bodies;
    }

    const std::vector<spaceship>& getShips() const {
        return ships;
    }

    void runSimulation(double totalTime, double timestep) {
        double currentTime = 0.0;
        int steps = 0;

        std::cout << "Starting simulation with timestep: " << timestep << " seconds" << std::endl;

        while (currentTime < totalTime) {
            // Update environmental conditions
            updateEnvironment();

            // Check for anomalies
            detectAnomalies();

            // Update mission status
            updateMissionStatus();

            // Update captain data packet
            updateCaptainData(currentTime);

            // Send data to AI server
            aiClient.sendCaptainData(captainData);

            // Get control actions from AI
            aiClient.receiveControlAction(ships);

            // Update physics for one timestep
            engine.update(bodies, ships, timestep);

            currentTime += timestep;
            steps++;

            if (steps % 10 == 0) {
                std::cout << "Simulation progress: " << (currentTime / totalTime) * 100.0
                          << "% (" << currentTime << " / " << totalTime << " seconds)" << std::endl;
            }
        }

        std::cout << "Simulation completed after " << steps << " steps" << std::endl;
        saveTrajectories("trajectories.csv");
    }

    void saveTrajectories(const std::string& filename) {
        // Existing implementation...
    }
    
    ~SimulationManager() {
        engine.cleanupTrajectories();
    }
};

#endif //SIMULATION_MANAGER_CUH