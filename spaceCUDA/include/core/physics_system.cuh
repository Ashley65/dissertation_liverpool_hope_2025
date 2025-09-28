//
// Created by DevAccount on 27/03/2025.
//

#ifndef PHYSICS_SYSTEM_CUH
#define PHYSICS_SYSTEM_CUH
#pragma once

#include "ECS.cuh"
#include "physics_components.cuh"
#include "../physics/physics_engine.cuh"
#include "device_data_manager.cuh"

class PhysicsSystem {
private:
    EntityManager& entityManager;
    physics_engine& engine;
    DeviceDataManager& dataManager;

public:
    PhysicsSystem(EntityManager& em, physics_engine& eng, DeviceDataManager& dm)
        : entityManager(em), engine(eng), dataManager(dm) {}

    // Convert ECS entities to physics_engine Bodies
    std::vector<Body> generateBodies() {
        std::vector<Body> bodies;

        for (Entity e = 0; e < entityManager.nextEntity; e++) {
            auto transform = entityManager.getComponent<TransformComponent>(e);
            auto velocity = entityManager.getComponent<VelocityComponent>(e);
            auto accel = entityManager.getComponent<AccelerationComponent>(e);
            auto mass = entityManager.getComponent<MassComponent>(e);

            if (transform && velocity && mass) {
                Body body{};
                body.x = transform->x;
                body.y = transform->y;
                body.z = transform->z;
                body.vx = velocity->vx;
                body.vy = velocity->vy;
                body.vz = velocity->vz;

                // Add acceleration if present
                if (accel) {
                    body.ax = accel->ax;
                    body.ay = accel->ay;
                    body.az = accel->az;
                } else {
                    body.ax = body.ay = body.az = 0.0;
                }

                body.mass = mass->mass;
                body.radius = mass->radius;

                // Set additional flags
                auto celestial = entityManager.getComponent<CelestialBodyComponent>(e);
                body.isElastic = celestial ? (celestial->isElastic ? 1 : 0) : 0;

                auto ship = entityManager.getComponent<SpaceshipComponent>(e);
                body.isSpaceship = ship ? 1 : 0;

                bodies.push_back(body);
            }
        }

        return bodies;
    }

    // Generate spaceships specifically
    std::vector<spaceship> generateSpaceships() {
        std::vector<spaceship> ships;

        for (Entity e = 0; e < entityManager.nextEntity; e++) {
            auto shipComp = entityManager.getComponent<SpaceshipComponent>(e);
            if (!shipComp) continue;

            auto transform = entityManager.getComponent<TransformComponent>(e);
            auto velocity = entityManager.getComponent<VelocityComponent>(e);
            auto accel = entityManager.getComponent<AccelerationComponent>(e);
            auto mass = entityManager.getComponent<MassComponent>(e);

            if (transform && velocity && mass) {
                spaceship ship{};

                // Set base body properties
                ship.base.x = transform->x;
                ship.base.y = transform->y;
                ship.base.z = transform->z;
                ship.base.vx = velocity->vx;
                ship.base.vy = velocity->vy;
                ship.base.vz = velocity->vz;

                if (accel) {
                    ship.base.ax = accel->ax;
                    ship.base.ay = accel->ay;
                    ship.base.az = accel->az;
                } else {
                    ship.base.ax = ship.base.ay = ship.base.az = 0.0;
                }

                ship.base.mass = mass->mass;
                ship.base.radius = mass->radius;
                ship.base.isSpaceship = 1;
                ship.base.isElastic = 0;

                // Set spaceship-specific properties
                ship.fuel = shipComp->fuel;
                ship.thrust = shipComp->thrust;
                ship.maxThrust = shipComp->maxThrust;
                ship.dirX = shipComp->dirX;
                ship.dirY = shipComp->dirY;
                ship.dirZ = shipComp->dirZ;
                ship.angularVelocity = shipComp->angularVelocity;

                ships.push_back(ship);
            }
        }

        return ships;
    }

    // Update physics and write back to components
    void update(float deltaTime) {
        // Get bodies and ships from entities
        std::vector<Body> bodies = generateBodies();
        std::vector<spaceship> ships = generateSpaceships();

        // Upload data to GPU
        dataManager.uploadBodies(bodies);
        if (!ships.empty()) {
            dataManager.uploadShips(ships);
        }

        // Compute gravitational forces using your existing physics engine
        engine.computeForces(bodies);

        // Update positions and velocities using the computed forces
        engine.update(bodies, ships, deltaTime);

        // Download updated data back to CPU
        dataManager.downloadBodies(bodies);
        if (!ships.empty()) {
            dataManager.downloadShips(ships);
        }

        // Write back data to components
        updateComponentsFromBodies(bodies);
        updateComponentsFromShips(ships);
    }

    void updateComponentsFromBodies(const std::vector<Body>& bodies) {
        int bodyIndex = 0;

        for (Entity e = 0; e < entityManager.nextEntity; e++) {
            // Skip entities that are spaceships
            if (entityManager.getComponent<SpaceshipComponent>(e)) {
                continue;
            }

            auto transform = entityManager.getComponent<TransformComponent>(e);
            auto velocity = entityManager.getComponent<VelocityComponent>(e);
            auto accel = entityManager.getComponent<AccelerationComponent>(e);

            if (transform && velocity && bodyIndex < bodies.size()) {
                // Update position
                transform->x = bodies[bodyIndex].x;
                transform->y = bodies[bodyIndex].y;
                transform->z = bodies[bodyIndex].z;

                // Update velocity
                velocity->vx = bodies[bodyIndex].vx;
                velocity->vy = bodies[bodyIndex].vy;
                velocity->vz = bodies[bodyIndex].vz;

                // Update acceleration
                if (accel) {
                    accel->ax = bodies[bodyIndex].ax;
                    accel->ay = bodies[bodyIndex].ay;
                    accel->az = bodies[bodyIndex].az;
                }

                bodyIndex++;
            }
        }
    }

    void updateComponentsFromShips(const std::vector<spaceship>& ships) {
        int shipIndex = 0;

        for (Entity e = 0; e < entityManager.nextEntity; e++) {
            auto shipComp = entityManager.getComponent<SpaceshipComponent>(e);
            if (!shipComp || shipIndex >= ships.size()) continue;

            auto transform = entityManager.getComponent<TransformComponent>(e);
            auto velocity = entityManager.getComponent<VelocityComponent>(e);
            auto accel = entityManager.getComponent<AccelerationComponent>(e);

            if (transform && velocity) {
                // Update position
                transform->x = ships[shipIndex].base.x;
                transform->y = ships[shipIndex].base.y;
                transform->z = ships[shipIndex].base.z;

                // Update velocity
                velocity->vx = ships[shipIndex].base.vx;
                velocity->vy = ships[shipIndex].base.vy;
                velocity->vz = ships[shipIndex].base.vz;

                // Update acceleration
                if (accel) {
                    accel->ax = ships[shipIndex].base.ax;
                    accel->ay = ships[shipIndex].base.ay;
                    accel->az = ships[shipIndex].base.az;
                }

                // Update ship-specific properties
                shipComp->fuel = ships[shipIndex].fuel;
                shipComp->thrust = ships[shipIndex].thrust;
                shipComp->dirX = ships[shipIndex].dirX;
                shipComp->dirY = ships[shipIndex].dirY;
                shipComp->dirZ = ships[shipIndex].dirZ;
                shipComp->angularVelocity = ships[shipIndex].angularVelocity;

                shipIndex++;
            }
        }
    }
};

#endif //PHYSICS_SYSTEM_CUH
