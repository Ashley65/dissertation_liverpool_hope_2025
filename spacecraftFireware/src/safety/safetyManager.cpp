//
// Created by DevAccount on 30/04/2025.
//

#include <random>
#include <safety/safetyManager.h>

#include "logger/LogMacros.h"

size_t SafetyManager::getMemoryOffset(const std::string& key) const {
    return static_cast<size_t>(std::hash<std::string>{}(key) % MEMORY_SIZE);
}

void SafetyManager::writeToHealth(const std::string& key, const std::string& value) {
    std::string data = key + ":" + value;
    size_t offset = getMemoryOffset(key);
    healthData->write(offset, data.c_str(), data.size());
}

std::string SafetyManager::readFromHealth(const std::string& key) {
    char buffer[256];
    size_t offset = getMemoryOffset(key);
    size_t read = healthData->read(offset, buffer, sizeof(buffer));
    if (read > 0) {
        std::string data(buffer, read);
        size_t pos = data.find(':');
        if (pos != std::string::npos) {
            return data.substr(pos + 1);
        }
    }
    return "0.0";
}

SafetyManager::SafetyManager() {
    healthData = std::make_unique<shortTermMemory>();
    diagnosticLogs = std::make_unique<longTermMemory>();

    // Initialize simulated system parameters
    writeToHealth("system_temp", "35.0");
    writeToHealth("power_level", "100.0");
    writeToHealth("fuel_pressure", "95.0");
    writeToHealth("engine_status", "nominal");
    writeToHealth("thrust_level", "0.0");
    writeToHealth("radiation_level", "0.15");
    writeToHealth("navigation_status", "operational");
}

SafetyManager::~SafetyManager() {
    stop();
}

void SafetyManager::start() {
    if (!running) {
        running = true;
        monitorThread = std::thread(&SafetyManager::monitoringLoop, this);
        LOG_INFO("[Safety] Safety monitoring system started");
    }
}

void SafetyManager::stop() {
    if (running) {
        running = false;
        if (monitorThread.joinable()) {
            monitorThread.join();
        }
        LOG_INFO("[Safety] Safety monitoring system stopped");
    }
}

void SafetyManager::addSafetyFlag(const std::string& flag, bool value) {
    std::lock_guard<std::mutex> lock(safetyFlagsMutex);
    safetyFlags[flag] = value;
}
void SafetyManager::monitoringLoop() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> tempDist(-0.2, 0.3);
    std::uniform_real_distribution<> powerDist(-0.1, 0.0);

    while (running) {
        try {
            // System responsiveness check
            if (!checkSystemResponsiveness()) {
                consecutiveFailures++;
                if (consecutiveFailures >= MAX_FAILURES) {
                    triggerEmergencyProtocol("System unresponsive");
                }
            } else {
                consecutiveFailures = 0;
            }

            // Simulate system parameters
            double temp = std::stod(readFromHealth("system_temp"));
            double power = std::stod(readFromHealth("power_level"));
            double fuel = std::stod(readFromHealth("fuel_pressure"));

            // Update simulated values
            temp += tempDist(gen);
            power += powerDist(gen);
            fuel -= 0.01;

            writeToHealth("system_temp", std::to_string(temp));
            writeToHealth("power_level", std::to_string(power));
            writeToHealth("fuel_pressure", std::to_string(fuel));

            // Check for critical conditions
            if (temp > 75.0) {
                currentState = SafetyState::DEGRADED;
                LOG_WARNING("[Safety] High temperature detected: " + std::to_string(temp));
            }
            if (power < 30.0) {
                currentState = SafetyState::CRITICAL;
                LOG_WARNING("[Safety] Low power level: " + std::to_string(power));
            }
            if (fuel < FUEL_CRITICAL_THRESHOLD) {
                triggerEmergencyProtocol("Critical fuel level");
            }

            // Record diagnostic data
            std::string diagnosticEntry = "Status: Temp=" + std::to_string(temp) +
                                        " Power=" + std::to_string(power) +
                                        " Fuel=" + std::to_string(fuel);
            diagnosticLogs->write(0, diagnosticEntry.c_str(), diagnosticEntry.size());

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } catch (const std::exception& e) {
            LOG_ERROR("[Safety] Error in monitoring loop: " + std::string(e.what()));
            triggerEmergencyProtocol("Internal safety monitor error");
        }
    }
}

void SafetyManager::triggerEmergencyProtocol(const std::string& reason) {
    LOG_ERROR("[Safety] EMERGENCY PROTOCOL TRIGGERED: " + reason);
    currentState = SafetyState::EMERGENCY;

    // Store emergency event in diagnostic logs
    std::string diagnosticEntry = "EMERGENCY: " + reason + " at " +
        std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    diagnosticLogs->write(0, diagnosticEntry.c_str(), diagnosticEntry.size());

    // Execute emergency procedures
    haltThrust();
    switchToLowPowerMode();
    executeEmergencyFallback();
}

void SafetyManager::executeEmergencyFallback() {
    LOG_WARNING("[Safety] Executing emergency fallback procedures");

    writeToHealth("thrust_level", "0.0");
    writeToHealth("power_level", "25.0");
    writeToHealth("engine_status", "emergency_shutdown");
    writeToHealth("navigation_status", "emergency_mode");

    std::string fallbackLog = "Emergency fallback executed at " +
        std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    diagnosticLogs->write(1, fallbackLog.c_str(), fallbackLog.size());
}

void SafetyManager::switchToLowPowerMode() {
    LOG_WARNING("[Safety] Switching to low-power mode");

    double currentPower = std::stod(readFromHealth("power_level"));
    writeToHealth("power_level", std::to_string(currentPower * 0.25));
    writeToHealth("navigation_status", "minimal_power");
}

void SafetyManager::haltThrust() {
    LOG_WARNING("[Safety] Emergency thrust halt initiated");

    writeToHealth("thrust_level", "0.0");
    writeToHealth("engine_status", "halted");
}


bool SafetyManager::checkSystemResponsiveness() {
    auto start = std::chrono::high_resolution_clock::now();

    // Simulate system checks
    bool healthCheck = simulateHealthCheck();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
        (end - start).count();

    lastResponseTime = duration;
    return healthCheck && (duration < RESPONSE_TIMEOUT_MS);
}

bool SafetyManager::isSystemHealthy() const {
    return currentState == SafetyState::NOMINAL;
}

SafetyState SafetyManager::getCurrentState() const {
    return currentState;
}

void SafetyManager::evaluateEnvironmentalData(const json& environmentalData) {
    if (!environmentalData.is_object()) {
        LOG_WARNING("Invalid environmental data format");
        return;
    }

    // Check for dangerous radiation levels
    if (environmentalData.contains("radiation_level")) {
        float radiationLevel = environmentalData["radiation_level"];
        if (radiationLevel > 8.0) {
            LOG_WARNING("Critical radiation levels detected: " + std::to_string(radiationLevel));
            triggerEmergencyProtocol("High radiation exposure");
        } else if (radiationLevel > 5.0) {
            LOG_WARNING("Elevated radiation levels detected: " + std::to_string(radiationLevel));
            addSafetyFlag("radiation_warning", true);
        }
    }

    // Check for temperature anomalies
    if (environmentalData.contains("external_temperature")) {
        float temperature = environmentalData["external_temperature"];
        float tempThresholdHigh = 120.0f;  // Celsius
        float tempThresholdLow = -150.0f;  // Celsius

        if (temperature > tempThresholdHigh || temperature < tempThresholdLow) {
            LOG_WARNING("Temperature outside safe parameters: " + std::to_string(temperature) + "C");
            addSafetyFlag("temperature_warning", true);
        }
    }

    // Check for debris or collision risks
    if (environmentalData.contains("nearby_objects")) {
        auto nearbyObjects = environmentalData["nearby_objects"];
        if (nearbyObjects.is_array() && !nearbyObjects.empty()) {
            for (const auto& object : nearbyObjects) {
                if (object.contains("distance") && object.contains("relative_velocity")) {
                    float distance = object["distance"];
                    float relVelocity = object["relative_velocity"];

                    // Calculate collision risk based on distance and relative velocity
                    if (distance < 5.0 && relVelocity > 10.0) {
                        LOG_WARNING("High collision risk detected!");
                        triggerEmergencyProtocol("Imminent collision risk");
                        break;
                    } else if (distance < 20.0) {
                        LOG_WARNING("Object in close proximity detected");
                        addSafetyFlag("proximity_warning", true);
                    }
                }
            }
        }
    }

    // Check atmospheric conditions (if applicable)
    if (environmentalData.contains("atmospheric_conditions")) {
        auto atmosphere = environmentalData["atmospheric_conditions"];
        if (atmosphere.contains("turbulence") && atmosphere["turbulence"].is_number()) {
            float turbulence = atmosphere["turbulence"];
            if (turbulence > 7.0) {
                LOG_WARNING("Severe atmospheric turbulence detected");
                addSafetyFlag("atmospheric_warning", true);
            }
        }
    }

    // Update overall system health status based on environmental data
    evaluateSystemHealth();
}

bool SafetyManager::simulateHealthCheck() {
    double temp = std::stod(readFromHealth("system_temp"));
    double power = std::stod(readFromHealth("power_level"));
    double fuel = std::stod(readFromHealth("fuel_pressure"));

    return temp < 80.0 && power > 20.0 && fuel > FUEL_CRITICAL_THRESHOLD;
}
void SafetyManager::evaluateSystemHealth() {
    std::lock_guard<std::mutex> lock(safetyFlagsMutex);

    // Check if we're already in emergency state
    if (currentState == SafetyState::EMERGENCY) {
        return; // Don't downgrade from emergency state
    }

    // Count active safety warnings
    int activeWarnings = 0;
    for (const auto& [flag, active] : safetyFlags) {
        if (active) {
            activeWarnings++;
        }
    }

    // Read critical system parameters
    double temp = std::stod(readFromHealth("system_temp"));
    double power = std::stod(readFromHealth("power_level"));
    double fuel = std::stod(readFromHealth("fuel_pressure"));
    std::string engineStatus = readFromHealth("engine_status");

    // Determine system state based on conditions
    if (activeWarnings >= 3 ||
        temp > 70.0 ||
        power < 40.0 ||
        fuel < FUEL_WARNING_THRESHOLD ||
        engineStatus == "failing") {

        currentState = SafetyState::CRITICAL;
        LOG_WARNING("[Safety] System health is CRITICAL");
        }
    else if (activeWarnings >= 1 ||
             temp > 60.0 ||
             power < 60.0 ||
             fuel < FUEL_CAUTION_THRESHOLD) {

        currentState = SafetyState::DEGRADED;
        LOG_INFO("[Safety] System health is DEGRADED");
             }
    else {
        currentState = SafetyState::NOMINAL;
        LOG_DEBUG("[Safety] System health is NOMINAL");
    }

    // Record current health state
    std::string healthState;
    switch (currentState) {
        case SafetyState::NOMINAL: healthState = "NOMINAL"; break;
        case SafetyState::DEGRADED: healthState = "DEGRADED"; break;
        case SafetyState::CRITICAL: healthState = "CRITICAL"; break;
        case SafetyState::EMERGENCY: healthState = "EMERGENCY"; break;
    }

    writeToHealth("system_health", healthState);
}


void SafetyManager::simulateSystemChecks() {
    try {
        // Simulate temperature fluctuation
        double temp = std::stod(readFromHealth("system_temp"));
        temp += (std::rand() % 3 - 1) * 0.1; // Random fluctuation
        writeToHealth("system_temp", std::to_string(temp));

        // Simulate power drain
        double power = std::stod(readFromHealth("power_level"));
        power -= 0.01; // Gradual power drain
        writeToHealth("power_level", std::to_string(power));

        // Update system state based on simulated values
        if (temp > 70.0) {
            currentState = SafetyState::DEGRADED;
        } else if (power < 30.0) {
            currentState = SafetyState::CRITICAL;
        } else if (currentState != SafetyState::EMERGENCY) {
            currentState = SafetyState::NOMINAL;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("[Safety] Error in system checks simulation: " + std::string(e.what()));
    }
}