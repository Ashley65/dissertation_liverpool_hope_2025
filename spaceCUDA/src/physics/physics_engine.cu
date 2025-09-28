//
// Created by DevAccount on 04/03/2025.
//
#include "../include/physics/physics_engine.cuh"

// Device (GPU) constants
__device__ const float G = 6.67430e-11f;       // Gravitational constant
__device__ const float THETA = 0.5f;           // Barnes-Hut opening angle parameter

// Host (CPU) constants - same values but accessible from host code
const float G_HOST = 6.67430e-11f;             // Gravitational constant for host code
const float THETA_HOST = 0.5f;                 // Barnes-Hut parameter for host code

void physics_engine::update(std::vector<Body>& bodies, std::vector<spaceship>& ships, float deltaTime) {
    numBodies = bodies.size();
    numSpaceships = ships.size();

    // Copy bodies to the device
    cudaMemcpy(d_bodies, bodies.data(), numBodies * sizeof(Body), cudaMemcpyHostToDevice);

    if (numSpaceships > 0) {
        cudaMemcpy(d_spaceships, ships.data(), numSpaceships * sizeof(spaceship), cudaMemcpyHostToDevice);
    }

    // Choose between Barnes-Hut and naive approach
    if (useBarnesHut) {
        // Use Barnes-Hut optimization (CPU implementation)
        Octree* root = buildOctree(bodies);
        calculateMassDistribution(root);
        computeForcesBarnesHut(bodies, root);
        delete root;

        // Update device memory with calculated accelerations
        cudaMemcpy(d_bodies, bodies.data(), numBodies * sizeof(Body), cudaMemcpyHostToDevice);
    } else {
        // Compute forces using the existing derivatives kernel
        int threadsPerBlock = 256;
        int numBlocks = (numBodies + threadsPerBlock - 1) / threadsPerBlock;

        // Calculate derivatives (forces/accelerations)
        computeDerivatives<<<numBlocks, threadsPerBlock>>>(d_bodies, d_k1, numBodies);
        cudaDeviceSynchronize();
    }

    // Perform RK45 integration
    integrateRK45(d_bodies, fminf(deltaTime, 1.0f));

    // Record trajectory points if trajectories are initialized
    if (d_trajectories != nullptr) {
        int threadsPerBlock = 256;
        int numBlocks = (numBodies + threadsPerBlock - 1) / threadsPerBlock;
        recordTrajectoryPoints<<<numBlocks, threadsPerBlock>>>(
            d_bodies, d_trajectories, numBodies, deltaTime);
        cudaDeviceSynchronize();
    }

    // Copy results back to host
    cudaMemcpy(bodies.data(), d_bodies, numBodies * sizeof(Body), cudaMemcpyDeviceToHost);

    if (numSpaceships > 0) {
        cudaMemcpy(ships.data(), d_spaceships, numSpaceships * sizeof(spaceship), cudaMemcpyDeviceToHost);
    }
}

physics_engine::physics_engine(int maxBodies, int maxSpaceships)  : useBarnesHut(false), numBodies(maxBodies), numSpaceships(maxSpaceships) {
    // Allocate device memory for bodies

    cudaMalloc(&d_bodies, maxBodies * sizeof(Body));
    cudaMalloc(&d_k1, maxBodies * sizeof(Body));
    cudaMalloc(&d_k2, maxBodies * sizeof(Body));
    cudaMalloc(&d_k3, maxBodies * sizeof(Body));
    cudaMalloc(&d_k4, maxBodies * sizeof(Body));
    cudaMalloc(&d_k5, maxBodies * sizeof(Body));
    cudaMalloc(&d_k6, maxBodies * sizeof(Body));
    cudaMalloc(&d_temp, maxBodies * sizeof(Body));
    cudaMalloc(&d_spaceships, maxSpaceships * sizeof(spaceship));

}

physics_engine::~physics_engine() {
    // Free device memory
    cudaFree(d_bodies);
    cudaFree(d_k1);
    cudaFree(d_k2);
    cudaFree(d_k3);
    cudaFree(d_k4);
    cudaFree(d_k5);
    cudaFree(d_k6);
    cudaFree(d_temp);
    cudaFree(d_spaceships);
}

void physics_engine::initTrajectories(int numBodies, int maxPoints, float recordInterval) {
    // Allocate host-side trajectory structures
    h_trajectories = new Trajectory[numBodies];

    // Allocate device-side trajectories array
    cudaMalloc(&d_trajectories, numBodies * sizeof(Trajectory));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error allocating trajectory memory: %s\n", cudaGetErrorString(err));
        return;
    }

    // Allocate position and velocity storage pointers arrays
    d_positionsStorage = new float3*[numBodies];
    d_velocitiesStorage = new float3*[numBodies];

    for (int i = 0; i < numBodies; i++) {
        // Configure each trajectory
        h_trajectories[i].maxPoints = maxPoints;
        h_trajectories[i].currentSize = 0;
        h_trajectories[i].recordInterval = recordInterval;
        h_trajectories[i].timeSinceLastRecord = 0;

        // Allocate memory for trajectory positions on device
        cudaMalloc(&d_positionsStorage[i], maxPoints * sizeof(float3));
        h_trajectories[i].positions = d_positionsStorage[i];

        // Allocate memory for trajectory velocities on device
        cudaMalloc(&d_velocitiesStorage[i], maxPoints * sizeof(float3));
        h_trajectories[i].velocities = d_velocitiesStorage[i];
    }

    // Copy trajectory structures to device
    cudaMemcpy(d_trajectories, h_trajectories, numBodies * sizeof(Trajectory),
              cudaMemcpyHostToDevice);
}

__global__ void computeForcesKernel(Body* d_bodies, int numBodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;

    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    const float softening = 1.0e9f; // Same softening parameter

    for (int j = 0; j < numBodies; ++j) {
        if (i != j) {
            float dx = d_bodies[j].x - d_bodies[i].x;
            float dy = d_bodies[j].y - d_bodies[i].y;
            float dz = d_bodies[j].z - d_bodies[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + softening * softening;
            float dist = sqrtf(distSqr);
            float force = (G * d_bodies[i].mass * d_bodies[j].mass) / distSqr;

            ax += force * dx / (dist * d_bodies[i].mass);
            ay += force * dy / (dist * d_bodies[i].mass);
            az += force * dz / (dist * d_bodies[i].mass);
        }
    }

    d_bodies[i].ax = ax;
    d_bodies[i].ay = ay;
    d_bodies[i].az = az;
}

void physics_engine::cleanupTrajectories() {
    if (h_trajectories) {
        // Free all position and velocity arrays
        for (int i = 0; i < numBodies; i++) {
            if (d_positionsStorage[i]) {
                cudaFree(d_positionsStorage[i]);
            }
            if (d_velocitiesStorage[i]) {
                cudaFree(d_velocitiesStorage[i]);
            }
        }

        delete[] d_positionsStorage;
        delete[] d_velocitiesStorage;
        cudaFree(d_trajectories);
        delete[] h_trajectories;

        d_positionsStorage = nullptr;
        d_velocitiesStorage = nullptr;
        d_trajectories = nullptr;
        h_trajectories = nullptr;
    }
}

std::vector<std::vector<TrajectoryPoint>> physics_engine::getTrajectories() const {
    std::vector<std::vector<TrajectoryPoint>> result(numBodies);

    // Update host-side trajectories from device
    cudaMemcpy(h_trajectories, d_trajectories, numBodies * sizeof(Trajectory),
              cudaMemcpyDeviceToHost);

    // For each body
    for (int i = 0; i < numBodies; i++) {
        int size = h_trajectories[i].currentSize;
        result[i].resize(size);

        // Copy trajectory data from device to host
        if (size > 0) {
            // Temporary arrays to hold position and velocity data
            float3* positions = new float3[size];
            float3* velocities = new float3[size];

            cudaMemcpy(positions, h_trajectories[i].positions,
                      size * sizeof(float3), cudaMemcpyDeviceToHost);
            cudaMemcpy(velocities, h_trajectories[i].velocities,
                      size * sizeof(float3), cudaMemcpyDeviceToHost);

            // Combine into trajectory points
            for (int j = 0; j < size; j++) {
                result[i][j].position = positions[j];
                result[i][j].velocity = velocities[j];
            }

            delete[] positions;
            delete[] velocities;
        }
    }

    return result;
}

// Launch CUDA kernel
void physics_engine::computeForces(std::vector<Body>& bodies) {
    int numBodies = bodies.size();
    Body* d_bodies;

    if (cudaMalloc(&d_bodies, numBodies * sizeof(Body)) != cudaSuccess) {
        printf("CUDA malloc failed!\n");
        return;
    }

    cudaMemcpy(d_bodies, bodies.data(), numBodies * sizeof(Body), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (numBodies + threadsPerBlock - 1) / threadsPerBlock;
    computeForcesKernel<<<numBlocks, threadsPerBlock>>>(d_bodies, numBodies);

    cudaMemcpy(bodies.data(), d_bodies, numBodies * sizeof(Body), cudaMemcpyDeviceToHost);
    cudaFree(d_bodies);
}

// Integrate motion (basic Euler method for now)
__global__ void integrateKernel(Body* d_bodies, int numBodies, float deltaTime) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;

    d_bodies[i].vx += d_bodies[i].ax * deltaTime;
    d_bodies[i].vy += d_bodies[i].ay * deltaTime;
    d_bodies[i].vz += d_bodies[i].az * deltaTime;

    d_bodies[i].x += d_bodies[i].vx * deltaTime;
    d_bodies[i].y += d_bodies[i].vy * deltaTime;
    d_bodies[i].z += d_bodies[i].vz * deltaTime;
}

__global__ void updateSpaceships(spaceship* ships, int numShips, float deltaTime) {

}


Octree * physics_engine::buildOctree(const std::vector<Body> &bodies) {
    // Find bounding box
    float minX = FLT_MAX, minY = FLT_MAX, minZ = FLT_MAX;
    float maxX = -FLT_MAX, maxY = -FLT_MAX, maxZ = -FLT_MAX;

    for (const auto& body : bodies) {
        minX = min(minX, body.x);
        minY = min(minY, body.y);
        minZ = min(minZ, body.z);
        maxX = max(maxX, body.x);
        maxY = max(maxY, body.y);
        maxZ = max(maxZ, body.z);
    }

    // Add some margin
    float margin = max(max(maxX - minX, maxY - minY), maxZ - minZ) * 0.1f;
    minX -= margin; minY -= margin; minZ -= margin;
    maxX += margin; maxY += margin; maxZ += margin;

    // Create root node with size that encompasses all bodies
    float size = max(max(maxX - minX, maxY - minY), maxZ - minZ);
    float centerX = (minX + maxX) / 2;
    float centerY = (minY + maxY) / 2;
    float centerZ = (minZ + maxZ) / 2;

    Octree* root = new Octree(centerX, centerY, centerZ, size);

    // Insert all bodies
    for (const auto& body : bodies) {
        insertBody(root, body);
    }

    // Calculate center of mass for all nodes
    calculateMassDistribution(root);

    return root;
}

void physics_engine::insertBody(Octree* node, const Body& body) {
    // If this node is a leaf and doesn't contain a body yet
    if (node->isLeaf && node->body == nullptr) {
        node->body = new Body(body);
        return;
    }

    // If this node is a leaf but already contains a body, split it
    if (node->isLeaf && node->body != nullptr) {
        Body* oldBody = node->body;
        node->body = nullptr;
        node->isLeaf = false;

        // Create children nodes if they don't exist
        createChildren(node);

        // Determine which child the existing body belongs to
        insertBody(node, *oldBody);
        delete oldBody;
    }

    // Determine which child the new body belongs to
    int octant = getOctant(node, body.x, body.y, body.z);

    // Create the child if it doesn't exist
    if (node->children[octant] == nullptr) {
        createChild(node, octant);
    }

    // Insert the body into the appropriate child
    insertBody(node->children[octant], body);
}

void physics_engine::createChildren(Octree *node) {
    float halfSize = node->size / 2;

    for (int i = 0; i < 8; i++) {
        if (node->children[i] == nullptr) {
            createChild(node, i);
        }
    }
}

void physics_engine::createChild(Octree *node, int octant) {
    float halfSize = node->size / 2;
    float newSize = node->size / 2;

    // Determine the new center based on the octant
    float offsetX = ((octant & 1) ? halfSize : -halfSize) / 2;
    float offsetY = ((octant & 2) ? halfSize : -halfSize) / 2;
    float offsetZ = ((octant & 4) ? halfSize : -halfSize) / 2;

    float newX = node->x + offsetX;
    float newY = node->y + offsetY;
    float newZ = node->z + offsetZ;

    node->children[octant] = new Octree(newX, newY, newZ, newSize);
}

int physics_engine::getOctant(const Octree *node, float x, float y, float z) {
    int octant = 0;
    if (x >= node->x) octant |= 1;
    if (y >= node->y) octant |= 2;
    if (z >= node->z) octant |= 4;
    return octant;
}

void physics_engine::computeForcesBarnesHut(std::vector<Body>& bodies, Octree* root) {
    // Process each body and calculate gravitational forces
    for (int i = 0; i < bodies.size(); i++) {
        // Reset acceleration
        bodies[i].ax = 0.0f;
        bodies[i].ay = 0.0f;
        bodies[i].az = 0.0f;

        // Compute force vector for this body
        Vec4 force = {0.0f, 0.0f, 0.0f, 0.0f};

        // Calculate force from octree
        computeForceFromOctree(root, bodies[i], force);

        // Convert force to acceleration (F = ma, so a = F/m)
        bodies[i].ax = force.x / bodies[i].mass;
        bodies[i].ay = force.y / bodies[i].mass;
        bodies[i].az = force.z / bodies[i].mass;

        // Safety check to prevent NaN values
        if (std::isnan(bodies[i].ax) || std::isnan(bodies[i].ay) || std::isnan(bodies[i].az)) {
            bodies[i].ax = 0.0f;
            bodies[i].ay = 0.0f;
            bodies[i].az = 0.0f;

            // Log warning about NaN values
            //printf("Warning: NaN detected in acceleration calculation for body %d\n", i);
        }
    }
}


void physics_engine::computeForceFromOctree(Octree *node, const Body &body, Vec4 &force) {
    if (!node || node->mass <= 0) return;

    // Skip self-interaction for leaf nodes containing the same body
    if (node->isLeaf && node->body != nullptr) {
        if (abs(node->x - body.x) < 1e-6f &&
            abs(node->y - body.y) < 1e-6f &&
            abs(node->z - body.z) < 1e-6f) {
            return;
            }
    }

    float dx = node->x - body.x;
    float dy = node->y - body.y;
    float dz = node->z - body.z;
    float distSqr = dx*dx + dy*dy + dz*dz;

    // Enhanced safety factor
    const float safetyFactor = 1.0e10f;  // Increased from 1.0e9f

    // Prevent division by zero or very small numbers
    if (distSqr < 1.0e-10f) {
        return; // Skip very close interactions
    }

    // Add softening only for close interactions
    if (distSqr < safetyFactor * safetyFactor) {
        distSqr += safetyFactor * safetyFactor;
    }

    float distance = sqrtf(distSqr);

    // Barnes-Hut approximation criterion
    if (node->isLeaf || (node->size / distance) < THETA_HOST) {
        // Calculate gravitational force with distance cube for numerical stability
        float invDist3 = 1.0f / (distSqr * distance);
        float f = G_HOST * body.mass * node->mass * invDist3;

        // Apply force limit for stability
        const float maxForce = 1.0e18f;  // Slightly reduced from 1.0e20f
        f = fminf(f, maxForce);

        force.x += f * dx;
        force.y += f * dy;
        force.z += f * dz;
    } else {
        // Recursively compute forces from children
        for (int i = 0; i < 8; i++) {
            if (node->children[i] != nullptr) {
                computeForceFromOctree(node->children[i], body, force);
            }
        }
    }
}

void physics_engine::calculateMassDistribution(Octree *node) {
    if (node == nullptr) return;

    if (node->isLeaf && node->body != nullptr) {
        node->mass = node->body->mass;
        node->x = node->body->x;
        node->y = node->body->y;
        node->z = node->body->z;
        return;
    }

    node->mass = 0;
    float massX = 0, massY = 0, massZ = 0;

    for (int i = 0; i < 8; i++) {
        if (node->children[i] != nullptr) {
            calculateMassDistribution(node->children[i]);

            node->mass += node->children[i]->mass;
            massX += node->children[i]->x * node->children[i]->mass;
            massY += node->children[i]->y * node->children[i]->mass;
            massZ += node->children[i]->z * node->children[i]->mass;
        }
    }

    if (node->mass > 0) {
        node->x = massX / node->mass;
        node->y = massY / node->mass;
        node->z = massZ / node->mass;
    }
}

void physics_engine::flattenOctree(Octree *root, std::vector<float4> &positions, std::vector<float> &masses,
    std::vector<int> &children) {
}

void physics_engine::integrateRK45(Body *d_bodies, float dt) const {
    int threadsPerBlock = 256;
    int numBlocks = (numBodies + threadsPerBlock - 1) / threadsPerBlock;

    // For now, use the naive O(n²) gravitational force calculation
    // In a full implementation, this would be replaced by Barnes-Hut

    // k1 = f(y_n)
    computeDerivatives<<<numBlocks, threadsPerBlock>>>(d_bodies, d_k1, numBodies);

    // k2 = f(y_n + dt/4 * k1)
    computeState<<<numBlocks, threadsPerBlock>>>(d_bodies, d_k1, d_temp, numBodies, dt/4.0f);
    computeDerivatives<<<numBlocks, threadsPerBlock>>>(d_temp, d_k2, numBodies);

    // k3 = f(y_n + 3dt/32 * k1 + 9dt/32 * k2)
    computeRK45Stage3<<<numBlocks, threadsPerBlock>>>(d_bodies, d_k1, d_k2, d_temp, numBodies, dt);
    computeDerivatives<<<numBlocks, threadsPerBlock>>>(d_temp, d_k3, numBodies);

    // k4 = f(y_n + 1932dt/2197 * k1 - 7200dt/2197 * k2 + 7296dt/2197 * k3)
    computeRK45Stage4<<<numBlocks, threadsPerBlock>>>(d_bodies, d_k1, d_k2, d_k3, d_temp, numBodies, dt);
    computeDerivatives<<<numBlocks, threadsPerBlock>>>(d_temp, d_k4, numBodies);

    // k5 = f(y_n + 439dt/216 * k1 - 8dt * k2 + 3680dt/513 * k3 - 845dt/4104 * k4)
    computeRK45Stage5<<<numBlocks, threadsPerBlock>>>(d_bodies, d_k1, d_k2, d_k3, d_k4, d_temp, numBodies, dt);
    computeDerivatives<<<numBlocks, threadsPerBlock>>>(d_temp, d_k5, numBodies);

    // k6 = f(y_n - 8dt/27 * k1 + 2dt * k2 - 3544dt/2565 * k3 + 1859dt/4104 * k4 - 11dt/40 * k5)
    computeRK45Stage6<<<numBlocks, threadsPerBlock>>>(d_bodies, d_k1, d_k2, d_k3, d_k4, d_k5, d_temp, numBodies, dt);
    computeDerivatives<<<numBlocks, threadsPerBlock>>>(d_temp, d_k6, numBodies);

    // y_{n+1} = y_n + dt * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)
    updateStateRK45<<<numBlocks, threadsPerBlock>>>(d_bodies, d_k1, d_k3, d_k4, d_k5, d_k6, numBodies, dt);
}


__global__ void computeBarnesHutForcesKernel(Body* bodies, float4* nodePositions, float* nodeMasses,
                                             int* nodeChildren, int numBodies, int numNodes) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numBodies) return;

        float ax = 0.0f, ay = 0.0f, az = 0.0f;

        // Stack-based tree traversal (avoids recursion)
        int stack[64];  // Maximum depth of octree
        int stackSize = 0;
        stack[stackSize++] = 0;  // Start with root node

        while (stackSize > 0) {
            int nodeIdx = stack[--stackSize];

            float4 nodePos = nodePositions[nodeIdx];
            float nodeMass = nodeMasses[nodeIdx];
            float nodeSize = nodePos.w;  // Use w component for size

            float dx = nodePos.x - bodies[i].x;
            float dy = nodePos.y - bodies[i].y;
            float dz = nodePos.z - bodies[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + 1e-6f;

            // Check if this is a leaf node or if it's far enough (Barnes-Hut criterion)
            if (nodeChildren[nodeIdx * 8] < 0 || (nodeSize / sqrtf(distSqr) < THETA)) {
                // Don't apply force if this is the same body (only relevant for leaves)
                if (distSqr > 1e-10f) {
                    float dist = sqrtf(distSqr);
                    float force = (G * bodies[i].mass * nodeMass) / distSqr;

                    ax += force * dx / (dist * bodies[i].mass);
                    ay += force * dy / (dist * bodies[i].mass);
                    az += force * dz / (dist * bodies[i].mass);
                }
            } else {
                // Add children to stack for processing
                for (int j = 0; j < 8; j++) {
                    int childIdx = nodeChildren[nodeIdx * 8 + j];
                    if (childIdx >= 0) {
                        stack[stackSize++] = childIdx;
                    }
                }
            }
        }

        bodies[i].ax = ax;
        bodies[i].ay = ay;
        bodies[i].az = az;

        // If this is a spaceship, add thrust from engine (will be handled separately)
        if (bodies[i].isSpaceship == 1) {
            // No need to add anything here since we handle thrust in updateSpaceships
        }
}


__global__ void computeDerivatives(Body* bodies, Body* derivatives, int numBodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;

    // Copy position and velocity
    derivatives[i].x = bodies[i].vx;
    derivatives[i].y = bodies[i].vy;
    derivatives[i].z = bodies[i].vz;

    // Calculate accelerations with improved softening
    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    // Softening parameter - much larger for astronomical scales
    const float softening = 1.0e9f; // 1 million km softening

    for (int j = 0; j < numBodies; ++j) {
        if (i != j) {
            float dx = bodies[j].x - bodies[i].x;
            float dy = bodies[j].y - bodies[i].y;
            float dz = bodies[j].z - bodies[i].z;

            // Use softened distance formula
            float distSqr = dx * dx + dy * dy + dz * dz + softening * softening;
            float dist = sqrtf(distSqr);
            float force = (G * bodies[i].mass * bodies[j].mass) / distSqr;

            ax += force * dx / (dist * bodies[i].mass);
            ay += force * dy / (dist * bodies[i].mass);
            az += force * dz / (dist * bodies[i].mass);
        }
    }

    derivatives[i].vx = ax;
    derivatives[i].vy = ay;
    derivatives[i].vz = az;
    derivatives[i].mass = bodies[i].mass;
    derivatives[i].radius = bodies[i].radius;
    derivatives[i].isSpaceship = bodies[i].isSpaceship;
    derivatives[i].isElastic = bodies[i].isElastic;
}

__global__ void recordTrajectoryPoints(Body* bodies, Trajectory* trajectories, int numBodies, float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;

    Trajectory& traj = trajectories[idx];

    // Only record if we've reached the recording interval
    traj.timeSinceLastRecord += deltaTime;
    if (traj.timeSinceLastRecord >= traj.recordInterval) {
        // Reset timer
        traj.timeSinceLastRecord = 0.0f;

        // Get current point index and increment with wraparound
        int pointIdx = traj.currentSize % traj.maxPoints;

        // Record the position and velocity
        traj.positions[pointIdx] = make_float3(bodies[idx].x, bodies[idx].y, bodies[idx].z);
        traj.velocities[pointIdx] = make_float3(bodies[idx].vx, bodies[idx].vy, bodies[idx].vz);

        // Update the number of recorded points
        if (traj.currentSize < traj.maxPoints) {
            traj.currentSize++;
        }
    }
}


__global__ void computeState(Body* y0, Body* k, Body* result, int numBodies, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;

    result[i].x = y0[i].x + dt * k[i].x;
    result[i].y = y0[i].y + dt * k[i].y;
    result[i].z = y0[i].z + dt * k[i].z;
    result[i].vx = y0[i].vx + dt * k[i].vx;
    result[i].vy = y0[i].vy + dt * k[i].vy;
    result[i].vz = y0[i].vz + dt * k[i].vz;
    result[i].mass = y0[i].mass;
    result[i].radius = y0[i].radius;
    result[i].isSpaceship = y0[i].isSpaceship;
    result[i].isElastic = y0[i].isElastic;

}



__global__ void computeRK45Stage3(Body* y0, Body* k1, Body* k2, Body* result, int numBodies, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;

    result[i].x = y0[i].x + dt * (3.0f/32.0f * k1[i].x + 9.0f/32.0f * k2[i].x);
    result[i].y = y0[i].y + dt * (3.0f/32.0f * k1[i].y + 9.0f/32.0f * k2[i].y);
    result[i].z = y0[i].z + dt * (3.0f/32.0f * k1[i].z + 9.0f/32.0f * k2[i].z);
    result[i].vx = y0[i].vx + dt * (3.0f/32.0f * k1[i].vx + 9.0f/32.0f * k2[i].vx);
    result[i].vy = y0[i].vy + dt * (3.0f/32.0f * k1[i].vy + 9.0f/32.0f * k2[i].vy);
    result[i].vz = y0[i].vz + dt * (3.0f/32.0f * k1[i].vz + 9.0f/32.0f * k2[i].vz);
    result[i].mass = y0[i].mass;
    result[i].radius = y0[i].radius;
    result[i].isSpaceship = y0[i].isSpaceship;
    result[i].isElastic = y0[i].isElastic;
}


__global__ void computeRK45Stage4(Body* y0, Body* k1, Body* k2, Body* k3, Body* result, int numBodies, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;

    result[i].x = y0[i].x + dt * (1932.0f/2197.0f * k1[i].x - 7200.0f/2197.0f * k2[i].x + 7296.0f/2197.0f * k3[i].x);
    result[i].y = y0[i].y + dt * (1932.0f/2197.0f * k1[i].y - 7200.0f/2197.0f * k2[i].y + 7296.0f/2197.0f * k3[i].y);
    result[i].z = y0[i].z + dt * (1932.0f/2197.0f * k1[i].z - 7200.0f/2197.0f * k2[i].z + 7296.0f/2197.0f * k3[i].z);
    result[i].vx = y0[i].vx + dt * (1932.0f/2197.0f * k1[i].vx - 7200.0f/2197.0f * k2[i].vx + 7296.0f/2197.0f * k3[i].vx);
    result[i].vy = y0[i].vy + dt * (1932.0f/2197.0f * k1[i].vy - 7200.0f/2197.0f * k2[i].vy + 7296.0f/2197.0f * k3[i].vy);
    result[i].vz = y0[i].vz + dt * (1932.0f/2197.0f * k1[i].vz - 7200.0f/2197.0f * k2[i].vz + 7296.0f/2197.0f * k3[i].vz);
    result[i].mass = y0[i].mass;
    result[i].radius = y0[i].radius;
    result[i].isSpaceship = y0[i].isSpaceship;
    result[i].isElastic = y0[i].isElastic;

}



__global__ void computeRK45Stage5(Body* y0, Body* k1, Body* k2, Body* k3, Body* k4, Body* result, int numBodies, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;

    result[i].x = y0[i].x + dt * (439.0f/216.0f * k1[i].x - 8.0f * k2[i].x + 3680.0f/513.0f * k3[i].x - 845.0f/4104.0f * k4[i].x);
    result[i].y = y0[i].y + dt * (439.0f/216.0f * k1[i].y - 8.0f * k2[i].y + 3680.0f/513.0f * k3[i].y - 845.0f/4104.0f * k4[i].y);
    result[i].z = y0[i].z + dt * (439.0f/216.0f * k1[i].z - 8.0f * k2[i].z + 3680.0f/513.0f * k3[i].z - 845.0f/4104.0f * k4[i].z);
    result[i].vx = y0[i].vx + dt * (439.0f/216.0f * k1[i].vx - 8.0f * k2[i].vx + 3680.0f/513.0f * k3[i].vx - 845.0f/4104.0f * k4[i].vx);
    result[i].vy = y0[i].vy + dt * (439.0f/216.0f * k1[i].vy - 8.0f * k2[i].vy + 3680.0f/513.0f * k3[i].vy - 845.0f/4104.0f * k4[i].vy);
    result[i].vz = y0[i].vz + dt * (439.0f/216.0f * k1[i].vz - 8.0f * k2[i].vz + 3680.0f/513.0f * k3[i].vz - 845.0f/4104.0f * k4[i].vz);
    result[i].mass = y0[i].mass;
    result[i].radius = y0[i].radius;
    result[i].isSpaceship = y0[i].isSpaceship;
    result[i].isElastic = y0[i].isElastic;
}


__global__ void computeRK45Stage6(Body* y0, Body* k1, Body* k2, Body* k3, Body* k4, Body* k5, Body* result, int numBodies, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;

    result[i].x = y0[i].x + dt * (-8.0f/27.0f * k1[i].x + 2.0f * k2[i].x - 3544.0f/2565.0f * k3[i].x + 1859.0f/4104.0f * k4[i].x - 11.0f/40.0f * k5[i].x);
    result[i].y = y0[i].y + dt * (-8.0f/27.0f * k1[i].y + 2.0f * k2[i].y - 3544.0f/2565.0f * k3[i].y + 1859.0f/4104.0f * k4[i].y - 11.0f/40.0f * k5[i].y);
    result[i].z = y0[i].z + dt * (-8.0f/27.0f * k1[i].z + 2.0f * k2[i].z - 3544.0f/2565.0f * k3[i].z + 1859.0f/4104.0f * k4[i].z - 11.0f/40.0f * k5[i].z);
    result[i].vx = y0[i].vx + dt * (-8.0f/27.0f * k1[i].vx + 2.0f * k2[i].vx - 3544.0f/2565.0f * k3[i].vx + 1859.0f/4104.0f * k4[i].vx - 11.0f/40.0f * k5[i].vx);
    result[i].vy = y0[i].vy + dt * (-8.0f/27.0f * k1[i].vy + 2.0f * k2[i].vy - 3544.0f/2565.0f * k3[i].vy + 1859.0f/4104.0f * k4[i].vy - 11.0f/40.0f * k5[i].vy);
    result[i].vz = y0[i].vz + dt * (-8.0f/27.0f * k1[i].vz + 2.0f * k2[i].vz - 3544.0f/2565.0f * k3[i].vz + 1859.0f/4104.0f * k4[i].vz - 11.0f/40.0f * k5[i].vz);
    result[i].mass = y0[i].mass;
    result[i].radius = y0[i].radius;
    result[i].isSpaceship = y0[i].isSpaceship;
    result[i].isElastic = y0[i].isElastic;
}



__global__ void updateStateRK45(Body* bodies, Body* k1, Body* k3, Body* k4, Body* k5, Body* k6, int numBodies, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;

    // Update position and velocity using RK45 weights
    bodies[i].x += dt * (16.0f/135.0f * k1[i].x + 6656.0f/12825.0f * k3[i].x + 28561.0f/56430.0f * k4[i].x - 9.0f/50.0f * k5[i].x + 2.0f/55.0f * k6[i].x);
    bodies[i].y += dt * (16.0f/135.0f * k1[i].y + 6656.0f/12825.0f * k3[i].y + 28561.0f/56430.0f * k4[i].y - 9.0f/50.0f * k5[i].y + 2.0f/55.0f * k6[i].y);
    bodies[i].z += dt * (16.0f/135.0f * k1[i].z + 6656.0f/12825.0f * k3[i].z + 28561.0f/56430.0f * k4[i].z - 9.0f/50.0f * k5[i].z + 2.0f/55.0f * k6[i].z);

    bodies[i].vx += dt * (16.0f/135.0f * k1[i].vx + 6656.0f/12825.0f * k3[i].vx + 28561.0f/56430.0f * k4[i].vx - 9.0f/50.0f * k5[i].vx + 2.0f/55.0f * k6[i].vx);
    bodies[i].vy += dt * (16.0f/135.0f * k1[i].vy + 6656.0f/12825.0f * k3[i].vy + 28561.0f/56430.0f * k4[i].vy - 9.0f/50.0f * k5[i].vy + 2.0f/55.0f * k6[i].vy);
    bodies[i].vz += dt * (16.0f/135.0f * k1[i].vz + 6656.0f/12825.0f * k3[i].vz + 28561.0f/56430.0f * k4[i].vz - 9.0f/50.0f * k5[i].vz + 2.0f/55.0f * k6[i].vz);
}

__global__ void detectCollisionsKernel(Body* bodies, int* collisionFlags, int numBodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;

    collisionFlags[i] = 0; // Initialize collision flag

    for (int j = i + 1; j < numBodies; ++j) {
        float dx = bodies[j].x - bodies[i].x;
        float dy = bodies[j].y - bodies[i].y;
        float dz = bodies[j].z - bodies[i].z;
        float distSqr = dx * dx + dy * dy + dz * dz;

        // Check for collision based on distance and radii
        if (distSqr < (bodies[i].radius + bodies[j].radius) * (bodies[i].radius + bodies[j].radius)) {
            collisionFlags[i] = 1;
            collisionFlags[j] = 1; // Mark both bodies as colliding
        }
    }
}
// __global__ void applySpacecraftControlKernel(
//     spaceship* ships,
//     int numShips,
//     float dt
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numShips) return;
//
//     spaceship& ship = ships[idx];
//
//     // Calculate thrust acceleration
//     float thrustAccelX = ship.thrustX / ship.mass;
//     float thrustAccelY = ship.thrustY / ship.mass;
//     float thrustAccelZ = ship.thrustZ / ship.mass;
//
//     // Apply thrust acceleration to velocity
//     ship.vx += thrustAccelX * dt;
//     ship.vy += thrustAccelY * dt;
//     ship.vz += thrustAccelZ * dt;
//
//     // Reduce fuel based on thrust magnitude
//     float thrustMagnitude = sqrtf(
//         ship.thrustX * ship.thrustX +
//         ship.thrustY * ship.thrustY +
//         ship.thrustZ * ship.thrustZ
//     );
//
//     // Simple fuel consumption model
//     const float fuelEfficiency = 1.0e-6f; // kg of fuel per Newton-second
//     ship.fuel -= thrustMagnitude * dt * fuelEfficiency;
//
//     // Clamp fuel to non-negative values
//     ship.fuel = fmaxf(ship.fuel, 0.0f);
// }


