#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include "../include/physics/physics_engine.cuh"
#include "../include/physics/Body.cuh"
#include "../include/core/ECS.cuh"
#include "../include/core/physics_components.cuh"
#include "../include/core/physics_system.cuh"
#include "../include/core/device_data_manager.cuh"
#include "core/dataChecker.cuh"
// Define astronomical unit (AU) in meters
const double AU = 1.496e11;

// Function to create a planet with circular orbit
// Function to create a planet with proper orbital elements
Body createPlanet(const std::string& name, double mass, double radius, double distance, double eccentricity = 0.0, double inclination = 0.0) {
    Body planet;

    // Set basic properties
    planet.mass = mass;
    planet.radius = radius;
    planet.isSpaceship = 0;
    planet.isElastic = 0;

    // Calculate position based on orbital elements
    double a = distance;  // Semi-major axis
    // Start at perihelion (closest approach)
    planet.x = a * (1.0 - eccentricity);
    planet.y = 0.0;
    planet.z = 0.0;

    // Calculate orbital velocity
    double sunMass = 1.989e30;  // kg
    double mu = G * sunMass;

    // Velocity at perihelion is perpendicular to position vector
    double vel = std::sqrt(mu * ((2.0 / distance) - (1.0 / a)));

    planet.vx = 0.0;
    planet.vy = vel * std::cos(inclination);
    planet.vz = vel * std::sin(inclination);

    // Initialize accelerations to zero
    planet.ax = 0.0;
    planet.ay = 0.0;
    planet.az = 0.0;

    return planet;
}

// Calculate kinetic energy of the system
float calculateKineticEnergy(const std::vector<Body>& bodies) {
    float energy = 0.0f;
    for (const auto& body : bodies) {
        float v_squared = body.vx*body.vx + body.vy*body.vy + body.vz*body.vz;
        energy += 0.5f * body.mass * v_squared;
    }
    return energy;
}

// Calculate potential energy of the system
float calculatePotentialEnergy(const std::vector<Body>& bodies) {
    float energy = 0.0f;
    for (size_t i = 0; i < bodies.size(); i++) {
        for (size_t j = i+1; j < bodies.size(); j++) {
            float dx = bodies[j].x - bodies[i].x;
            float dy = bodies[j].y - bodies[i].y;
            float dz = bodies[j].z - bodies[i].z;
            float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
            energy -= G * bodies[i].mass * bodies[j].mass / distance;
        }
    }
    return energy;
}

// Calculate angular momentum of the system
// Calculate angular momentum of the system
void calculateAngularMomentum(const std::vector<Body>& bodies, float& Lx, float& Ly, float& Lz) {
    Lx = Ly = Lz = 0.0f;

    // Calculate relative to the origin for a solar system simulation
    // (since the Sun is dominant and nearly at the origin)
    for (const auto& body : bodies) {
        // L = r × p = m(r × v)
        Lx += body.mass * (body.y * body.vz - body.z * body.vy);
        Ly += body.mass * (body.z * body.vx - body.x * body.vz);
        Lz += body.mass * (body.x * body.vy - body.y * body.vx);
    }
}

// Save trajectories to a CSV file
void saveTrajectoriesToFile(const std::vector<std::vector<TrajectoryPoint>>& trajectories,
                           const char* planetNames[]) {
    std::ofstream file("trajectories.csv");

    // Write CSV header
    file << "Body,PointIndex,X,Y,Z,VX,VY,VZ\n";

    // Check if any trajectories exist
    if (trajectories.empty()) {
        std::cerr << "No trajectory data available to save" << std::endl;
        return;
    }

    // Write trajectory data
    for (size_t i = 0; i < trajectories.size(); i++) {
        std::cout << "Body " << i << " (" << planetNames[i] << ") has "
                  << trajectories[i].size() << " trajectory points" << std::endl;

        for (size_t j = 0; j < trajectories[i].size(); j++) {
            file << planetNames[i] << "," << j << ","
                 << trajectories[i][j].position.x << ","
                 << trajectories[i][j].position.y << ","
                 << trajectories[i][j].position.z << ","
                 << trajectories[i][j].velocity.x << ","
                 << trajectories[i][j].velocity.y << ","
                 << trajectories[i][j].velocity.z << "\n";
        }
    }

    file.close();
    std::cout << "Trajectories saved to trajectories.csv" << std::endl;
}

// In main.cu
// In main.cu
int main() {
    std::cout << "Starting Solar System Simulation with ECS" << std::endl;

    // Initialize EntityManager
    EntityManager entityManager;

    // Initialize physics engine
    int maxBodies = 9;  // Sun + 8 planets
    physics_engine engine(maxBodies, 0);  // No spaceships

    // Initialize PhysicsSystem
    PhysicsSystem physicsSystem(entityManager, engine, maxBodies);

    // Create Sun
    Entity sunEntity = entityManager.createEntity();
    entityManager.addComponent<TransformComponent>(sunEntity, 0.0, 0.0, 0.0);
    entityManager.addComponent<VelocityComponent>(sunEntity, 0.0, 0.0, 0.0);
    entityManager.addComponent<MassComponent>(sunEntity, 1.989e30f, 6.957e8f);
    entityManager.addComponent<CelestialBodyComponent>(sunEntity, false);
    entityManager.addComponent<TrajectoryComponent>(sunEntity, 1000, 86400.0f * 3);

    // Create Mercury
    Entity mercuryEntity = entityManager.createEntity();
    double mercuryDist = 0.387 * AU;
    double mercuryVel = std::sqrt((G * 1.989e30) / mercuryDist);
    entityManager.addComponent<TransformComponent>(mercuryEntity, mercuryDist, 0.0, 0.0);
    entityManager.addComponent<VelocityComponent>(mercuryEntity, 0.0, mercuryVel, 0.0);
    entityManager.addComponent<MassComponent>(mercuryEntity, 3.3011e23f, 2.4397e6f);
    entityManager.addComponent<CelestialBodyComponent>(mercuryEntity, false);
    entityManager.addComponent<TrajectoryComponent>(mercuryEntity, 1000, 86400.0f);

    // Create Venus
    Entity venusEntity = entityManager.createEntity();
    double venusDist = 0.723 * AU;
    double venusVel = std::sqrt((G * 1.989e30) / venusDist);
    entityManager.addComponent<TransformComponent>(venusEntity, venusDist, 0.0, 0.0);
    entityManager.addComponent<VelocityComponent>(venusEntity, 0.0, venusVel, 0.0);
    entityManager.addComponent<MassComponent>(venusEntity, 4.8675e24f, 6.0518e6f);
    entityManager.addComponent<CelestialBodyComponent>(venusEntity, false);
    entityManager.addComponent<TrajectoryComponent>(venusEntity, 1000, 86400.0f);

    // Create Earth
    Entity earthEntity = entityManager.createEntity();
    double earthDist = 1.0 * AU;
    double earthVel = std::sqrt((G * 1.989e30) / earthDist);
    entityManager.addComponent<TransformComponent>(earthEntity, earthDist, 0.0, 0.0);
    entityManager.addComponent<VelocityComponent>(earthEntity, 0.0, earthVel, 0.0);
    entityManager.addComponent<MassComponent>(earthEntity, 5.972e24f, 6.371e6f);
    entityManager.addComponent<CelestialBodyComponent>(earthEntity, false);
    entityManager.addComponent<TrajectoryComponent>(earthEntity, 1000, 86400.0f);

    // Create Mars
    Entity marsEntity = entityManager.createEntity();
    double marsDist = 1.524 * AU;
    double marsVel = std::sqrt((G * 1.989e30) / marsDist);
    entityManager.addComponent<TransformComponent>(marsEntity, marsDist, 0.0, 0.0);
    entityManager.addComponent<VelocityComponent>(marsEntity, 0.0, marsVel, 0.0);
    entityManager.addComponent<MassComponent>(marsEntity, 6.4171e23f, 3.3895e6f);
    entityManager.addComponent<CelestialBodyComponent>(marsEntity, false);
    entityManager.addComponent<TrajectoryComponent>(marsEntity, 1000, 86400.0f);

    // Create Jupiter
    Entity jupiterEntity = entityManager.createEntity();
    double jupiterDist = 5.203 * AU;
    double jupiterVel = std::sqrt((G * 1.989e30) / jupiterDist);
    entityManager.addComponent<TransformComponent>(jupiterEntity, jupiterDist, 0.0, 0.0);
    entityManager.addComponent<VelocityComponent>(jupiterEntity, 0.0, jupiterVel, 0.0);
    entityManager.addComponent<MassComponent>(jupiterEntity, 1.898e27f, 6.9911e7f);
    entityManager.addComponent<CelestialBodyComponent>(jupiterEntity, false);
    entityManager.addComponent<TrajectoryComponent>(jupiterEntity, 1000, 86400.0f * 7);

    // Create Saturn
    Entity saturnEntity = entityManager.createEntity();
    double saturnDist = 9.582 * AU;
    double saturnVel = std::sqrt((G * 1.989e30) / saturnDist);
    entityManager.addComponent<TransformComponent>(saturnEntity, saturnDist, 0.0, 0.0);
    entityManager.addComponent<VelocityComponent>(saturnEntity, 0.0, saturnVel, 0.0);
    entityManager.addComponent<MassComponent>(saturnEntity, 5.683e26f, 5.8232e7f);
    entityManager.addComponent<CelestialBodyComponent>(saturnEntity, false);
    entityManager.addComponent<TrajectoryComponent>(saturnEntity, 1000, 86400.0f * 14);

    // Create Uranus
    Entity uranusEntity = entityManager.createEntity();
    double uranusDist = 19.22 * AU;
    double uranusVel = std::sqrt((G * 1.989e30) / uranusDist);
    entityManager.addComponent<TransformComponent>(uranusEntity, uranusDist, 0.0, 0.0);
    entityManager.addComponent<VelocityComponent>(uranusEntity, 0.0, uranusVel, 0.0);
    entityManager.addComponent<MassComponent>(uranusEntity, 8.681e25f, 2.5362e7f);
    entityManager.addComponent<CelestialBodyComponent>(uranusEntity, false);
    entityManager.addComponent<TrajectoryComponent>(uranusEntity, 1000, 86400.0f * 30);

    // Create Neptune
    Entity neptuneEntity = entityManager.createEntity();
    double neptuneDist = 30.05 * AU;
    // Fix Neptune's velocity calculation
    double neptuneVel = std::sqrt(G * 1.989e30 / neptuneDist);
    entityManager.addComponent<TransformComponent>(neptuneEntity, neptuneDist, 0.0, 0.0);
    entityManager.addComponent<VelocityComponent>(neptuneEntity, 0.0, neptuneVel, 0.0);
    entityManager.addComponent<MassComponent>(neptuneEntity, 1.024e26f, 2.4622e7f);
    entityManager.addComponent<CelestialBodyComponent>(neptuneEntity, false);
    entityManager.addComponent<TrajectoryComponent>(neptuneEntity, 1000, 86400.0f * 30);

    // Initialize trajectories in the physics engine
    engine.initTrajectories(maxBodies, 1000, 86400.0f);

    // Simulation parameters
    float timeStep = 7200.0f;         // 2 hours in seconds
    int numSteps = 4380;              // Still simulate for 1 year
    int saveFrequency = 24;           // Save data every 24 hours

    // Create individual trajectory files
    std::ofstream mercuryFile("Mercury_trajectory.csv");
    std::ofstream venusFile("Venus_trajectory.csv");
    std::ofstream earthFile("Earth_trajectory.csv");
    std::ofstream marsFile("Mars_trajectory.csv");
    std::ofstream jupiterFile("Jupiter_trajectory.csv");
    std::ofstream saturnFile("Saturn_trajectory.csv");
    std::ofstream uranusFile("Uranus_trajectory.csv");
    std::ofstream neptuneFile("Neptune_trajectory.csv");

    // Write headers to each file
    mercuryFile << "x,y,z,vx,vy,vz\n";
    venusFile << "x,y,z,vx,vy,vz\n";
    earthFile << "x,y,z,vx,vy,vz\n";
    marsFile << "x,y,z,vx,vy,vz\n";
    jupiterFile << "x,y,z,vx,vy,vz\n";
    saturnFile << "x,y,z,vx,vy,vz\n";
    uranusFile << "x,y,z,vx,vy,vz\n";
    neptuneFile << "x,y,z,vx,vy,vz\n";

    // Run simulation
    for (int step = 0; step < numSteps; step++) {
        physicsSystem.update(timeStep);

        if (step % saveFrequency == 0) {
            std::cout << "Simulation day: " << step/24 << std::endl;

            // Get current positions and velocities
            std::vector<Body> bodies = physicsSystem.generateBodies();

            // Save planet positions to individual files
            if (bodies.size() >= 9) {
                // Mercury (index 1, Sun is index 0)
                mercuryFile << bodies[1].x << "," << bodies[1].y << "," << bodies[1].z << ","
                          << bodies[1].vx << "," << bodies[1].vy << "," << bodies[1].vz << "\n";

                // Venus (index 2)
                venusFile << bodies[2].x << "," << bodies[2].y << "," << bodies[2].z << ","
                         << bodies[2].vx << "," << bodies[2].vy << "," << bodies[2].vz << "\n";

                // Earth (index 3)
                earthFile << bodies[3].x << "," << bodies[3].y << "," << bodies[3].z << ","
                         << bodies[3].vx << "," << bodies[3].vy << "," << bodies[3].vz << "\n";

                // Mars (index 4)
                marsFile << bodies[4].x << "," << bodies[4].y << "," << bodies[4].z << ","
                        << bodies[4].vx << "," << bodies[4].vy << "," << bodies[4].vz << "\n";

                // Jupiter (index 5)
                jupiterFile << bodies[5].x << "," << bodies[5].y << "," << bodies[5].z << ","
                           << bodies[5].vx << "," << bodies[5].vy << "," << bodies[5].vz << "\n";

                // Saturn (index 6)
                saturnFile << bodies[6].x << "," << bodies[6].y << "," << bodies[6].z << ","
                          << bodies[6].vx << "," << bodies[6].vy << "," << bodies[6].vz << "\n";

                // Uranus (index 7)
                uranusFile << bodies[7].x << "," << bodies[7].y << "," << bodies[7].z << ","
                          << bodies[7].vx << "," << bodies[7].vy << "," << bodies[7].vz << "\n";

                // Neptune (index 8)
                neptuneFile << bodies[8].x << "," << bodies[8].y << "," << bodies[8].z << ","
                           << bodies[8].vx << "," << bodies[8].vy << "," << bodies[8].vz << "\n";
            }
        }
    }

    // Close individual files
    mercuryFile.close();
    venusFile.close();
    earthFile.close();
    marsFile.close();
    jupiterFile.close();
    saturnFile.close();
    uranusFile.close();
    neptuneFile.close();

    // Convert Neptune trajectory to AU and analyze
    std::vector<PlanetData> neptuneData = PlanetData::readFromFile("Neptune_trajectory.csv");
    std::cout << "\nAnalyzing Neptune's trajectory data..." << std::endl;
    std::cout << "Neptune trajectory data points: " << neptuneData.size() << std::endl;
    analyseTrajectoryData(neptuneData);




    // Get and save trajectories
    auto trajectories = engine.getTrajectories();
    const char* planetNames[] = {"Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"};

    // Save trajectories to file
    std::ofstream file("trajectories.csv");
    file << "Body,PointIndex,X,Y,Z,VX,VY,VZ\n";

    for (size_t i = 0; i < trajectories.size(); i++) {
        for (size_t j = 0; j < trajectories[i].size(); j++) {
            file << planetNames[i] << "," << j << ","
                 << trajectories[i][j].position.x << ","
                 << trajectories[i][j].position.y << ","
                 << trajectories[i][j].position.z << ","
                 << trajectories[i][j].velocity.x << ","
                 << trajectories[i][j].velocity.y << ","
                 << trajectories[i][j].velocity.z << "\n";
        }
    }
    file.close();

    // Calculate final energy and angular momentum
    std::vector<Body> finalBodies = physicsSystem.generateBodies();
    float kineticEnergy = 0.0f;
    float potentialEnergy = 0.0f;

    for (const auto& body : finalBodies) {
        float v_squared = body.vx*body.vx + body.vy*body.vy + body.vz*body.vz;
        kineticEnergy += 0.5f * body.mass * v_squared;
    }

    for (size_t i = 0; i < finalBodies.size(); i++) {
        for (size_t j = i+1; j < finalBodies.size(); j++) {
            float dx = finalBodies[j].x - finalBodies[i].x;
            float dy = finalBodies[j].y - finalBodies[i].y;
            float dz = finalBodies[j].z - finalBodies[i].z;
            float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
            potentialEnergy -= G * finalBodies[i].mass * finalBodies[j].mass / distance;
        }
    }

    float totalEnergy = kineticEnergy + potentialEnergy;

    std::cout << "Final simulation energy: " << totalEnergy << " J" << std::endl;
    std::cout << "Kinetic energy: " << kineticEnergy << " J" << std::endl;
    std::cout << "Potential energy: " << potentialEnergy << " J" << std::endl;

    engine.cleanupTrajectories();
    std::cout << "Simulation complete" << std::endl;
    return 0;
}