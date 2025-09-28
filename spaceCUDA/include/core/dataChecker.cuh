//
// Created by DevAccount on 28/03/2025.
//

#ifndef DATACHECKER_CUH
#define DATACHECKER_CUH
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>



// Structure to hold trajectory data points for analysis
struct PlanetData {
    float x, y, z;     // Position in meters
    float vx, vy, vz;  // Velocity in m/s

    // Calculate distance from the sun (origin)
    float distanceFromSun() const {
        return std::sqrt(x*x + y*y + z*z);
    }

    // Calculate speed in m/s
    float speed() const {
        return std::sqrt(vx*vx + vy*vy + vz*vz);
    }

    // Convert data from CSV file
    static std::vector<PlanetData> readFromFile(const std::string& filename) {
        std::vector<PlanetData> data;
        std::ifstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return data;
        }

        // Skip header if present
        std::string line;
        getline(file, line);

        // Read data rows
        while (getline(file, line)) {
            PlanetData point;
            std::istringstream iss(line);
            char comma;
            iss >> point.x >> comma >> point.y >> comma >> point.z >> comma
                >> point.vx >> comma >> point.vy >> comma >> point.vz;

            data.push_back(point);
        }

        return data;
    }
};

// Analyze trajectory data to find orbital characteristics
inline void analyseTrajectoryData(const std::vector<PlanetData>& data) {
    if (data.empty()) {
        std::cout << "No data to analyze." << std::endl;
        return;
    }

    // Convert to AU for easier interpretation
    const float AU = 1.496e11f;  // meters

    // Find perihelion and aphelion
    auto minmaxDistances = std::minmax_element(data.begin(), data.end(),
                                             [](const PlanetData& a, const PlanetData& b) {
                                                 return a.distanceFromSun() < b.distanceFromSun();
                                             });

    float minDist = minmaxDistances.first->distanceFromSun() / AU;
    float maxDist = minmaxDistances.second->distanceFromSun() / AU;

    // Calculate semi-major axis
    float semiMajorAxis = (minDist + maxDist) / 2.0f;

    // Calculate eccentricity
    float eccentricity = (maxDist - minDist) / (maxDist + minDist);

    // Calculate average, min, and max speed
    float avgSpeed = 0.0f;
    float minSpeed = FLT_MAX;
    float maxSpeed = 0.0f;

    for (const auto& point : data) {
        float speed = point.speed();
        avgSpeed += speed;
        minSpeed = std::min(minSpeed, speed);
        maxSpeed = std::max(maxSpeed, speed);
    }
    avgSpeed /= data.size();

    // Output results
    std::cout << "Orbital analysis results:" << std::endl;
    std::cout << "  Perihelion: " << minDist << " AU" << std::endl;
    std::cout << "  Aphelion: " << maxDist << " AU" << std::endl;
    std::cout << "  Semi-major axis: " << semiMajorAxis << " AU" << std::endl;
    std::cout << "  Eccentricity: " << eccentricity << std::endl;
    std::cout << "  Average speed: " << avgSpeed / 1000.0f << " km/s" << std::endl;
    std::cout << "  Min speed: " << minSpeed / 1000.0f << " km/s" << std::endl;
    std::cout << "  Max speed: " << maxSpeed / 1000.0f << " km/s" << std::endl;
}
#endif //DATACHECKER_CUH
