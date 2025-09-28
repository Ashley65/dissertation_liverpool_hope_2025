//
// Created by DevAccount on 04/03/2025.
//

#ifndef PHYSICS_ENGINE_CUH
#define PHYSICS_ENGINE_CUH
#include <cuda_runtime.h>
#include <vector>
#include "Body.cuh"

struct Vec4 {
    float x, y, z, w;

    Vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    Vec4(float _x, float _y, float _z, float _w = 0.0f) : x(_x), y(_y), z(_z), w(_w) {}
};

struct TrajectoryPoint {
    float3 position;
    float3 velocity;
};


struct Octree {
    float x, y, z;        // Center of mass
    float mass;           // Total mass
    float size;           // Size of this cube
    Body* body;           // Pointer to body (only for leaf nodes)
    Octree* children[8];  // Octants
    bool isLeaf;          // Is this a leaf node?

    Octree(float x, float y, float z, float size)
        : x(x), y(y), z(z), size(size), mass(0), body(nullptr), isLeaf(true) {
        for (int i = 0; i < 8; i++) {
            children[i] = nullptr;
        }
    }

    ~Octree() {
        for (int i = 0; i < 8; i++) {
            delete children[i];
        }
    }
};

// CUDA kernel functions must be declared outside the class
__global__ void computeState(Body* y0, Body* k, Body* result, int numBodies, float dt);
__global__ void computeRK45Stage3(Body* y0, Body* k1, Body* k2, Body* result, int numBodies, float dt);
__global__ void computeRK45Stage4(Body* y0, Body* k1, Body* k2, Body* k3, Body* result, int numBodies, float dt);
__global__ void computeRK45Stage5(Body* y0, Body* k1, Body* k2, Body* k3, Body* k4, Body* result, int numBodies, float dt);
__global__ void computeRK45Stage6(Body* y0, Body* k1, Body* k2, Body* k3, Body* k4, Body* k5, Body* result, int numBodies, float dt);
__global__ void updateStateRK45(Body* bodies, Body* k1, Body* k3, Body* k4, Body* k5, Body* k6, int numBodies, float dt);
__global__ void computeDerivatives(Body* bodies, Body* derivatives, int numBodies);
// Add this to your kernel declarations in physics_engine.cuh
__global__ void computeForcesKernel(Body* bodies, int numBodies);
__global__ void recordTrajectoryPoints(Body* bodies, Trajectory* trajectories, int numBodies, float deltaTime);

class physics_engine {
    public:
        void update(std::vector<Body>& bodies, std::vector<spaceship>& ships, float deltaTime);
        physics_engine(int maxBodies, int maxSpaceships);
        ~physics_engine();
        void initTrajectories(int numBodies, int maxPoints, float recordInterval);
        void cleanupTrajectories();
        std::vector<std::vector<TrajectoryPoint>> getTrajectories() const;
        static void computeForces(std::vector<Body>& bodies);
        bool useBarnesHut;

    private:

        //void integrate(std::vector<Body>& bodies, float deltaTime);

        Octree* buildOctree(const std::vector<Body>& bodies);
        void insertBody(Octree* node, const Body& body);
        void createChildren(Octree* node);
        void createChild(Octree* node, int octant);

        int getOctant(const Octree* node, float x, float y, float z);

        void computeForcesBarnesHut(std::vector<Body>& bodies, Octree* root);

        // Recursive function to compute force from the Octree
        static void computeForceFromOctree(Octree* node, const Body& body, Vec4& force);


        static void calculateMassDistribution(Octree* node);

        static void flattenOctree(Octree* root, std::vector<float4>& positions, std::vector<float>& masses, std::vector<int>& children);
        void integrateRK45(Body* d_bodies, float dt) const;

        // Device memory
        Body* d_bodies;
        Body* d_k1;
        Body* d_k2;
        Body* d_k3;
        Body* d_k4;
        Body* d_k5;
        Body* d_k6;
        Body* d_temp;
        spaceship* d_spaceships;
        int numSpaceships;
        int numBodies;

        // Add these new members
        Trajectory* h_trajectories;  // Host copy of trajectory data
        Trajectory* d_trajectories;  // Device trajectories data
        float3** d_positionsStorage; // Device storage for all trajectory points
        float3** d_velocitiesStorage;
};
#endif //PHYSICS_ENGINE_CUH
