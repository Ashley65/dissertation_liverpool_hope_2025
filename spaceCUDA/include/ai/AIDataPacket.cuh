#ifndef CAPTAIN_DATA_PACKET_CUH
#define CAPTAIN_DATA_PACKET_CUH
#pragma once

#include <string>
#include <vector>
#include "../physics/Body.cuh"

// Mission phases
enum class MissionPhase {
    LAUNCH,
    TRANSIT,
    EXPLORATION,
    CRITICAL,
    EMERGENCY
};

// Waypoint structure
struct Waypoint {
    float x, y, z;
    std::string name;
    bool reached;
};

// Anomaly information
struct AnomalyInfo {
    bool present;
    std::string type;
    float severity;
    float confidence;
};

// Memory record structure
struct MemoryRecord {
    std::string context;
    float relevance;
    std::string experience;
    float utilization;
};

// Performance metrics
struct PerformanceMetrics {
    float reward;
    float pathEfficiency;
    float fuelEfficiency;
    float anomalyHandlingScore;
};

// Environmental readings
struct EnvironmentalReadings {
    float temperature;
    float radiation;
    float magneticField;
    std::vector<std::string> hazards;
};

class CaptainDataPacket {
private:
    // Spacecraft position data
    float posX, posY, posZ;
    float waypointDistance;
    float targetDistance;  // Distance to Mars
    
    // Spacecraft state
    float velocityX, velocityY, velocityZ;
    float speed;
    float fuel;
    float missionTime;
    
    // Mission context
    MissionPhase phase;
    std::vector<Waypoint> waypoints;
    float missionProgress;  // 0.0 to 1.0
    
    // Environmental data
    EnvironmentalReadings environment;
    
    // Anomaly information
    AnomalyInfo anomaly;
    
    // Memory system
    std::vector<MemoryRecord> memories;
    
    // Performance metrics
    PerformanceMetrics performance;

public:
    // Constructor with default values
    CaptainDataPacket() : 
        posX(0), posY(0), posZ(0),
        waypointDistance(0), targetDistance(0),
        velocityX(0), velocityY(0), velocityZ(0),
        speed(0), fuel(100), missionTime(0),
        phase(MissionPhase::LAUNCH),
        missionProgress(0),
        anomaly({false, "", 0, 0}),
        performance({0, 0, 0, 0}) {}
    
    // Update from physics simulation data
    void updateFromSimulation(const std::vector<Body>& bodies, 
                              const std::vector<spaceship>& ships,
                              float currentTime) {
        if (ships.empty()) return;
        
        // Get ship data (assuming first ship is the captain's ship)
        const spaceship& ship = ships[0];
        
        // Update position data
        posX = ship.base.x;
        posY = ship.base.y;
        posZ = ship.base.z;
        
        // Update velocity data
        velocityX = ship.base.vx;
        velocityY = ship.base.vy;
        velocityZ = ship.base.vz;
        speed = std::sqrt(velocityX*velocityX + velocityY*velocityY + velocityZ*velocityZ);
        
        // Update fuel and mission time
        fuel = ship.fuel;
        missionTime = currentTime;
        
        // Find Mars in the bodies vector (assuming index 4 is Mars based on initialization)
        if (bodies.size() > 4) {
            const Body& mars = bodies[4]; // Mars is index 4 in the solar system initialization
            targetDistance = std::sqrt(
                std::pow(posX - mars.x, 2) +
                std::pow(posY - mars.y, 2) +
                std::pow(posZ - mars.z, 2)
            );
        }
        
        // Update waypoint distance if waypoints exist
        if (!waypoints.empty()) {
            const Waypoint& currentWaypoint = waypoints[0]; // Get the next unvisited waypoint
            waypointDistance = std::sqrt(
                std::pow(posX - currentWaypoint.x, 2) +
                std::pow(posY - currentWaypoint.y, 2) +
                std::pow(posZ - currentWaypoint.z, 2)
            );
        }
    }
    
    // Update mission context
    void updateMissionContext(MissionPhase newPhase, float progress, 
                              const std::vector<Waypoint>& newWaypoints) {
        phase = newPhase;
        missionProgress = progress;
        waypoints = newWaypoints;
    }
    
    // Update environmental readings
    void updateEnvironment(float temp, float rad, float magField, 
                           const std::vector<std::string>& hazardList) {
        environment.temperature = temp;
        environment.radiation = rad;
        environment.magneticField = magField;
        environment.hazards = hazardList;
    }
    
    // Update anomaly information
    void updateAnomaly(bool present, const std::string& type, 
                       float severity, float confidence) {
        anomaly.present = present;
        anomaly.type = type;
        anomaly.severity = severity;
        anomaly.confidence = confidence;
    }
    
    // Update memory system
    void updateMemories(const std::vector<MemoryRecord>& newMemories) {
        memories = newMemories;
    }
    
    // Update performance metrics
    void updatePerformance(float reward, float pathEff, 
                           float fuelEff, float anomalyScore) {
        performance.reward = reward;
        performance.pathEfficiency = pathEff;
        performance.fuelEfficiency = fuelEff;
        performance.anomalyHandlingScore = anomalyScore;
    }
    
    // Getters for all data sections
    
    // Position data
    float getPositionX() const { return posX; }
    float getPositionY() const { return posY; }
    float getPositionZ() const { return posZ; }
    float getWaypointDistance() const { return waypointDistance; }
    float getTargetDistance() const { return targetDistance; }
    
    // Spacecraft state
    float getVelocityX() const { return velocityX; }
    float getVelocityY() const { return velocityY; }
    float getVelocityZ() const { return velocityZ; }
    float getSpeed() const { return speed; }
    float getFuel() const { return fuel; }
    float getMissionTime() const { return missionTime; }
    
    // Mission context
    MissionPhase getPhase() const { return phase; }
    const std::vector<Waypoint>& getWaypoints() const { return waypoints; }
    float getMissionProgress() const { return missionProgress; }
    
    // Environmental data
    const EnvironmentalReadings& getEnvironment() const { return environment; }
    
    // Anomaly information
    const AnomalyInfo& getAnomaly() const { return anomaly; }
    
    // Memory system
    const std::vector<MemoryRecord>& getMemories() const { return memories; }
    
    // Performance metrics
    const PerformanceMetrics& getPerformance() const { return performance; }
};

#endif //CAPTAIN_DATA_PACKET_CUH