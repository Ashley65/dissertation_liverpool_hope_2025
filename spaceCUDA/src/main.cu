#include <iostream>
#include <chrono>
#include <string>
#include "../include/core/simulation_manager.cuh"

int main(int argc, char** argv) {
    // Default simulation parameters
    int maxBodies = 1000;
    int maxShips = 10;
    double totalSimTime = 86400.0 * 30.0; // 30 days in seconds
    double timeStep = 60.0; // 60 seconds per step
    std::string aiServerEndpoint = "tcp://localhost:5556";

    // Parse command line arguments if provided
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--time" && i + 1 < argc) {
                totalSimTime = std::stod(argv[++i]);
            } else if (arg == "--step" && i + 1 < argc) {
                timeStep = std::stod(argv[++i]);
            } else if (arg == "--bodies" && i + 1 < argc) {
                maxBodies = std::stoi(argv[++i]);
            } else if (arg == "--ships" && i + 1 < argc) {
                maxShips = std::stoi(argv[++i]);
            } else if (arg == "--server" && i + 1 < argc) {
                aiServerEndpoint = argv[++i];
            } else if (arg == "--help") {
                std::cout << "Solar System Simulation" << std::endl;
                std::cout << "Usage options:" << std::endl;
                std::cout << "  --time <seconds>   : Total simulation time in seconds (default: " << totalSimTime << ")" << std::endl;
                std::cout << "  --step <seconds>   : Time step in seconds (default: " << timeStep << ")" << std::endl;
                std::cout << "  --bodies <count>   : Maximum number of bodies (default: " << maxBodies << ")" << std::endl;
                std::cout << "  --ships <count>    : Maximum number of ships (default: " << maxShips << ")" << std::endl;
                std::cout << "  --server <endpoint>: AI server endpoint (default: " << aiServerEndpoint << ")" << std::endl;
                return 0;
            }
        }
    }

    try {
        std::cout << "Initializing simulation..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();

        // Initialize simulation manager
        SimulationManager simulator(maxBodies, maxShips, aiServerEndpoint);

        std::cout << "Running simulation for " << (totalSimTime / 86400.0)
                  << " days with " << timeStep << " second timesteps" << std::endl;

        // Run the simulation
        simulator.runSimulation(totalSimTime, timeStep);

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        std::cout << "Simulation completed in " << (duration.count() / 1000.0) << " seconds" << std::endl;
        std::cout << "Final state:" << std::endl;
        std::cout << "  Bodies: " << simulator.getBodies().size() << std::endl;
        std::cout << "  Ships: " << simulator.getShips().size() << std::endl;

        if (!simulator.getShips().empty()) {
            const auto& ship = simulator.getShips()[0];
            std::cout << "  Main spacecraft position: ["
                      << ship.x / 1.496e11f << " AU, "
                      << ship.y / 1.496e11f << " AU, "
                      << ship.z / 1.496e11f << " AU]" << std::endl;
            std::cout << "  Remaining fuel: " << ship.fuel << "%" << std::endl;
        }

        std::cout << "Trajectory data saved to trajectories.csv" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}