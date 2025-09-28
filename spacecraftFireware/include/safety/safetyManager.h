//
// Created by DevAccount on 30/04/2025.
//

#ifndef SAFETYMANAGER_H
#define SAFETYMANAGER_H
#pragma once
#include <packagem.h>

#include <memory/shortTermMemory.h>
#include <memory/longTermMemory.h>



enum class SafetyState {
    NOMINAL,
    DEGRADED,
    CRITICAL,
    EMERGENCY
};

class SafetyManager {
public:
    SafetyManager();
    ~SafetyManager();

    void start();
    void stop();
    void triggerEmergencyProtocol(const std::string& reason);
    bool isSystemHealthy() const;
    SafetyState getCurrentState() const;
    void evaluateEnvironmentalData(const json& environmentalData);

private:
    static constexpr size_t MEMORY_SIZE = 4096;
    static constexpr size_t MAX_FAILURES = 3;
    static constexpr double FUEL_CRITICAL_THRESHOLD = 10.0;
    static constexpr double RESPONSE_TIMEOUT_MS = 500.0;
    std::map<std::string, bool> safetyFlags;
    std::mutex safetyFlagsMutex;
    static constexpr double FUEL_CAUTION_THRESHOLD = 30.0;
    static constexpr double FUEL_WARNING_THRESHOLD = 15.0;

    void addSafetyFlag(const std::string& flag, bool value);

    void monitoringLoop();
    void executeEmergencyFallback();
    void switchToLowPowerMode();
    void haltThrust();
    bool checkSystemResponsiveness();
    bool simulateHealthCheck();

    void evaluateSystemHealth();

    void simulateSystemChecks();

    void writeToHealth(const std::string& key, const std::string& value);
    std::string readFromHealth(const std::string& key);
    size_t getMemoryOffset(const std::string& key) const;

    std::atomic<bool> running{false};
    std::atomic<SafetyState> currentState{SafetyState::NOMINAL};
    std::thread monitorThread;
    std::atomic<int> consecutiveFailures{0};
    std::atomic<double> lastResponseTime{0};

    std::unique_ptr<shortTermMemory> healthData;
    std::unique_ptr<longTermMemory> diagnosticLogs;
};




#endif //SAFETYMANAGER_H
