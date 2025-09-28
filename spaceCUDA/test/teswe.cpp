//
// Created by DevAccount on 28/03/2025.
//
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include "../include/physics/physics_engine.cuh"
#include "../include/physics/Body.cuh"

void ad() {
    std::cout << "Hello, World!" << std::endl;
}
// Helper function to calculate kinetic energy
float calculateKineticEnergy(const std::vector<Body>& bodies) {
    float energy = 0.0f;
    for (const auto& body : bodies) {
        float v_squared = body.vx * body.vx + body.vy * body.vy + body.vz * body.vz;
        energy += 0.5f * body.mass * v_squared;
    }
    return energy;
}

// Helper function to calculate potential energy
float calculatePotentialEnergy(const std::vector<Body>& bodies) {
    float energy = 0.0f;
    float G = 6.67430e-11f;
    float softening = 1.0e6f; // Softening parameter (1000 km)

    for (size_t i = 0; i < bodies.size(); ++i) {
        for (size_t j = i + 1; j < bodies.size(); ++j) {
            float dx = bodies[j].x - bodies[i].x;
            float dy = bodies[j].y - bodies[i].y;
            float dz = bodies[j].z - bodies[i].z;
            float r2 = dx*dx + dy*dy + dz*dz;

            if (r2 == 0.0f) {
                std::cerr << "Warning: Zero distance between bodies " << i << " and " << j << std::endl;
                r2 = softening * softening; // Use softening if distance is zero
            }

            float distance = std::sqrt(r2);
            energy -= G * bodies[i].mass * bodies[j].mass / distance;
        }
    }
    return energy;
}

// Helper function to calculate total angular momentum
void calculateAngularMomentum(const std::vector<Body>& bodies, float& Lx, float& Ly, float& Lz) {
    Lx = Ly = Lz = 0.0f;

    // Calculate angular momentum relative to center of mass
    float totalMass = 0.0f;
    float comX = 0.0f, comY = 0.0f, comZ = 0.0f;

    for (const auto& body : bodies) {
        totalMass += body.mass;
        comX += body.mass * body.x;
        comY += body.mass * body.y;
        comZ += body.mass * body.z;
    }

    comX /= totalMass;
    comY /= totalMass;
    comZ /= totalMass;

    for (const auto& body : bodies) {
        float rx = body.x - comX;
        float ry = body.y - comY;
        float rz = body.z - comZ;

        Lx += body.mass * (ry * body.vz - rz * body.vy);
        Ly += body.mass * (rz * body.vx - rx * body.vz);
        Lz += body.mass * (rx * body.vy - ry * body.vx);
    }
}

// Helper function to create a planet
Body createPlanet(const std::string& name, float mass, float radius, float distance, float inclination = 0.0f) {
    float G = 6.67430e-11f;
    Body planet;

    // Set basic properties
    planet.mass = mass;
    planet.radius = radius;
    planet.isSpaceship = 0;
    planet.isElastic = 0;

    // Calculate position (start all planets aligned along the x-axis, then rotate by inclination)
    planet.x = distance * std::cos(inclination);
    planet.y = 0.0f;
    planet.z = distance * std::sin(inclination);

    // Calculate orbital velocity for a circular orbit
    float sunMass = 1.989e30f;
    float v = std::sqrt(G * sunMass / distance);

    // Velocity perpendicular to position vector and in the orbital plane
    planet.vx = 0.0f;
    planet.vy = v * std::cos(inclination);
    planet.vz = 0.0f;

    std::cout << "Created " << name << " at distance " << distance/1.496e11f
              << " AU with velocity " << v/1000.0f << " km/s\n";

    return planet;
}
void saveTrajectoriesToFile(const std::vector<std::vector<TrajectoryPoint>>& trajectories, const char* planetNames[]) {
    for (size_t i = 0; i < trajectories.size(); i++) {
        std::string filename = std::string(planetNames[i]) + "_trajectory.csv";
        std::ofstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << std::endl;
            continue;
        }

        file << "x,y,z,vx,vy,vz\n";
        for (const auto& point : trajectories[i]) {
            file << point.position.x << ","
                 << point.position.y << ","
                 << point.position.z << ","
                 << point.velocity.x << ","
                 << point.velocity.y << ","
                 << point.velocity.z << "\n";
        }

        file.close();
        std::cout << "Saved trajectory for " << planetNames[i] << " to " << filename << std::endl;
    }
}

int main() {
    // Create solar system
    std::vector<Body> bodies;
    std::vector<spaceship> ships; // Empty for this test

    const float AU = 1.496e11f; // Astronomical Unit in meters

    // Sun at rest at the origin
    Body sun;
    sun.x = 0.0f;
    sun.y = 0.0f;
    sun.z = 0.0f;
    sun.vx = 0.0f;
    sun.vy = 0.0f;
    sun.vz = 0.0f;
    sun.mass = 1.989e30f;  // Sun's mass in kg
    sun.radius = 6.957e8f; // Sun's radius in m
    sun.isSpaceship = 0;
    sun.isElastic = 0;

    bodies.push_back(sun);

    // Add planets with approximate real values
    // Mercury
    bodies.push_back(createPlanet("Mercury", 3.3011e23f, 2.4397e6f, 0.387 * AU, 0.122f));

    // Venus
    bodies.push_back(createPlanet("Venus", 4.8675e24f, 6.0518e6f, 0.723 * AU, 0.059f));

    // Earth
    bodies.push_back(createPlanet("Earth", 5.972e24f, 6.371e6f, 1.0 * AU));

    // Mars
    bodies.push_back(createPlanet("Mars", 6.4171e23f, 3.3895e6f, 1.524 * AU, 0.032f));

    // Jupiter
    bodies.push_back(createPlanet("Jupiter", 1.8982e27f, 6.9911e7f, 5.2 * AU, 0.022f));

    // Saturn
    bodies.push_back(createPlanet("Saturn", 5.6834e26f, 5.8232e7f, 9.58 * AU, 0.043f));

    // Uranus
    bodies.push_back(createPlanet("Uranus", 8.6810e25f, 2.5362e7f, 19.2 * AU, 0.013f));

    // Neptune
    bodies.push_back(createPlanet("Neptune", 1.02413e26f, 2.4622e7f, 30.05 * AU, 0.011f));

    // Initialize physics engine
    physics_engine engine(bodies.size(), 0);


    // Initialize trajectories (e.g., record every 5 days, keep 1000 points per body)
    engine.initTrajectories(bodies.size(), 1000, 86400.0f * 5);

    // Calculate initial energy and angular momentum
    float initialKE = calculateKineticEnergy(bodies);
    float initialPE = calculatePotentialEnergy(bodies);
    float initialTotalEnergy = initialKE + initialPE;

    float initialLx, initialLy, initialLz;
    calculateAngularMomentum(bodies, initialLx, initialLy, initialLz);
    float initialAngularMomentumMag = std::sqrt(initialLx*initialLx + initialLy*initialLy + initialLz*initialLz);

    std::cout << "\nInitial conditions:" << std::endl;
    std::cout << "Initial kinetic energy: " << initialKE << " J" << std::endl;
    std::cout << "Initial potential energy: " << initialPE << " J" << std::endl;
    std::cout << "Initial total energy: " << initialTotalEnergy << " J" << std::endl;
    std::cout << "Initial angular momentum: " << initialAngularMomentumMag << " kg·m²/s" << std::endl;

    // Simulate for a period of time (e.g., 1 Earth year)
    float totalTime = 3.156e7f;  // 1 Earth year in seconds
    float dt = 86400.0f;  // 1-day timestep
    int steps = static_cast<int>(totalTime / dt);

    for (int step = 0; step < steps; ++step) {
        // Update the system
        engine.update(bodies, ships, dt);

        // At the end of the simulation, get and save trajectories
        std::vector<std::vector<TrajectoryPoint>> trajectories = engine.getTrajectories();


        // Print progress every 30 days
        if (step % 30 == 0) {
            float currentKE = calculateKineticEnergy(bodies);
            float currentPE = calculatePotentialEnergy(bodies);
            float currentTotalEnergy = currentKE + currentPE;

            float currentLx, currentLy, currentLz;
            calculateAngularMomentum(bodies, currentLx, currentLy, currentLz);
            float currentAngularMomentumMag = std::sqrt(currentLx*currentLx + currentLy*currentLy + currentLz*currentLz);

            float energyError = (currentTotalEnergy - initialTotalEnergy) / std::abs(initialTotalEnergy) * 100.0f;
            float momentumError = (currentAngularMomentumMag - initialAngularMomentumMag) / initialAngularMomentumMag * 100.0f;

            std::cout << "Day " << step << ":" << std::endl;
            std::cout << "  Earth position: (" << bodies[3].x/AU << ", " << bodies[3].y/AU << ", " << bodies[3].z/AU << ") AU" << std::endl;
            std::cout << "  Energy error: " << energyError << "%" << std::endl;
            std::cout << "  Angular momentum error: " << momentumError << "%" << std::endl;

        }
    }
    // At the end of the simulation, get and save trajectories
    std::vector<std::vector<TrajectoryPoint>> trajectories = engine.getTrajectories();

    // Final results
    std::cout << "\nFinal state after " << totalTime/86400.0f << " days:" << std::endl;

    const char* planetNames[] = {"Sun", "Mercury", "Venus", "Earth", "Mars",
                                "Jupiter", "Saturn", "Uranus", "Neptune"};

    for (size_t i = 0; i < bodies.size(); i++) {
        std::cout << planetNames[i] << " position: ("
                  << bodies[i].x/AU << ", "
                  << bodies[i].y/AU << ", "
                  << bodies[i].z/AU << ") AU" << std::endl;
    }

    float finalKE = calculateKineticEnergy(bodies);
    float finalPE = calculatePotentialEnergy(bodies);
    float finalTotalEnergy = finalKE + finalPE;

    float finalLx, finalLy, finalLz;
    calculateAngularMomentum(bodies, finalLx, finalLy, finalLz);
    float finalAngularMomentumMag = std::sqrt(finalLx*finalLx + finalLy*finalLy + finalLz*finalLz);

    float energyError = (finalTotalEnergy - initialTotalEnergy) / std::abs(initialTotalEnergy) * 100.0f;
    float momentumError = (finalAngularMomentumMag - initialAngularMomentumMag) / initialAngularMomentumMag * 100.0f;

    std::cout << "Final energy error: " << energyError << "%" << std::endl;
    std::cout << "Final angular momentum error: " << momentumError << "%" << std::endl;




    // Print trajectory stats


    // Save trajectories to file if desired
    saveTrajectoriesToFile(trajectories, planetNames);

    // Clean up
    engine.cleanupTrajectories();

    return 0;
}