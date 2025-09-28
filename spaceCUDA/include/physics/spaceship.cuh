#ifndef SPACESHIP_CUH
#define SPACESHIP_CUH

// Spaceship structure representing a spacecraft in the simulation
struct spaceshipBody {
    // Position in 3D space (meters)
    float x, y, z;

    // Velocity components (meters/second)
    float vx, vy, vz;

    // Mass (kg)
    float mass;

    // Fuel amount (kg)
    float fuel;

    // Thrust components (Newtons)
    float thrustX, thrustY, thrustZ;

    // Radius (meters)
    float radius;

    // Status flags
    bool isSpaceship;  // Always true for spaceships (for identification)
    bool isElastic;    // Whether the spaceship handles collisions elastically
};

// Structure to represent a trajectory point
// (This might be in a different file, but it's used with spacecraft)
// struct TrajectoryPoint {
//     struct {
//         float x, y, z;
//     } position;
//
//     struct {
//         float x, y, z;
//     } velocity;
// };

#endif // SPACESHIP_CUH
