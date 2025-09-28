//
// Created by DevAccount on 27/03/2025.
//

#ifndef PHYSICS_COMPONENTS_CUH
#define PHYSICS_COMPONENTS_CUH

#pragma once
#include "ECS.cuh"
#include <cuda_runtime.h>

#include "physics/physics_engine.cuh"

struct TransformComponent : public Component {
    double x, y, z;

    TransformComponent(double _x = 0, double _y = 0, double _z = 0)
        : x(_x), y(_y), z(_z) {}
};

struct VelocityComponent : public Component {
    double vx, vy, vz;

    VelocityComponent(double _vx = 0, double _vy = 0, double _vz = 0)
        : vx(_vx), vy(_vy), vz(_vz) {}
};

struct AccelerationComponent : public Component {
    double ax, ay, az;

    AccelerationComponent(double _ax = 0, double _ay = 0, double _az = 0)
        : ax(_ax), ay(_ay), az(_az) {}
};

struct MassComponent : public Component {
    double mass;
    double radius;

    MassComponent(double _mass = 0, double _radius = 0)
        : mass(_mass), radius(_radius) {}
};

struct CelestialBodyComponent : public Component {
    std::string name;
    bool isElastic;

    CelestialBodyComponent(const std::string& _name = "", bool _isElastic = false)
        : name(_name), isElastic(_isElastic) {}
};

struct SpaceshipComponent : public Component {
    double fuel;
    double thrust;
    double maxThrust;
    double dirX, dirY, dirZ;
    double angularVelocity;

    SpaceshipComponent(double _fuel = 0, double _maxThrust = 0)
        : fuel(_fuel), thrust(0), maxThrust(_maxThrust),
          dirX(0), dirY(0), dirZ(0), angularVelocity(0) {}
};

#endif //PHYSICS_COMPONENTS_CUH
