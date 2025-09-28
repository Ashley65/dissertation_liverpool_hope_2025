//
// Created by DevAccount on 04/03/2025.
//

#ifndef BODY_H
#define BODY_H
#include <cstdint>

struct Body {
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
    double mass;
    double radius;
    uint8_t isSpaceship;  // use 0 or 1
    uint8_t isElastic;    // use 0 or 1
};


struct spaceship {
    int id;
    Body base;
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
    // Inherits basic physics properties
    double fuel;              // Remaining fuel
    double thrust;            // Current thrust power
    double maxThrust;         // Maximum thrust power
    double dirX, dirY, dirZ;  // Orientation for thrust direction
    double angularVelocity;   // Rotational speed for turns
    double mass;             // Mass of the spaceship
    double radius;           // Radius of the spaceship
    double thrustX, thrustY, thrustZ; // Thrust components in 3D space
};

struct Trajectory {
    float3* positions;        // Device pointer to position history
    float3* velocities;       // Device pointer to velocity history
    int maxPoints;            // Maximum number of points to store
    int currentSize;          // Current number of stored points
    float recordInterval;     // Time interval between recordings
    float timeSinceLastRecord; // Time accumulator

    Trajectory() : positions(nullptr), velocities(nullptr), maxPoints(0), currentSize(0),
                   recordInterval(0.0f), timeSinceLastRecord(0.0f) {}
};

#endif //BODY_H
