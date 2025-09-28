//
// Created by DevAccount on 27/03/2025.
//

#ifndef DEVICE_DATA_MANAGER_CUH
#define DEVICE_DATA_MANAGER_CUH
#pragma once
#include <cuda_runtime.h>
#include "../physics/Body.cuh"
#include "ECS.cuh"


class DeviceDataManager {
private:
    Body* d_bodies;
    spaceship* d_ships;
    int maxEntities;

public:
    DeviceDataManager(int _maxEntities) : maxEntities(_maxEntities), d_bodies(nullptr), d_ships(nullptr) {
        // Allocate device memory
        cudaMalloc(&d_bodies, maxEntities * sizeof(Body));
        cudaMalloc(&d_ships, maxEntities * sizeof(spaceship));
    }

    ~DeviceDataManager() {
        if (d_bodies) cudaFree(d_bodies);
        if (d_ships) cudaFree(d_ships);
    }

    void uploadBodies(const std::vector<Body>& bodies) {
        if (!bodies.empty()) {
            cudaMemcpy(d_bodies, bodies.data(), bodies.size() * sizeof(Body), cudaMemcpyHostToDevice);
        }
    }

    void uploadShips(const std::vector<spaceship>& ships) {
        if (!ships.empty()) {
            cudaMemcpy(d_ships, ships.data(), ships.size() * sizeof(spaceship), cudaMemcpyHostToDevice);
        }
    }

    void downloadBodies(std::vector<Body>& bodies) {
        if (!bodies.empty()) {
            cudaMemcpy(bodies.data(), d_bodies, bodies.size() * sizeof(Body), cudaMemcpyDeviceToHost);
        }
    }

    void downloadShips(std::vector<spaceship>& ships) {
        if (!ships.empty()) {
            cudaMemcpy(ships.data(), d_ships, ships.size() * sizeof(spaceship), cudaMemcpyDeviceToHost);
        }
    }

    Body* getDeviceBodies() { return d_bodies; }
    spaceship* getDeviceShips() { return d_ships; }
};

#endif //DEVICE_DATA_MANAGER_CUH
