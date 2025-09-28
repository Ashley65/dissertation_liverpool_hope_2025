//
// Created by DevAccount on 04/04/2025.
//

#ifndef PACKAGEM_H
#define PACKAGEM_H
#pragma once

#include <zmq.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <thread>
#include <mutex>
#include <chrono>
#include <functional>
#include <nlohmann/json.hpp>
#include <fstream>

#include <ctime>
#include <cmath>
#include <atomic>
#include <memory>
#include <algorithm>

using json = nlohmann::json;
using namespace std::chrono;



// Enums
enum class MissionPhase {
    LAUNCH,
    TRANSIT,
    EXPLORATION,
    RETURN,
    LANDING,
    EMERGENCY
};

enum class CommandPriority {
    EMERGENCY = 0,
    HIGH = 1,
    NORMAL = 2,
    LOW = 3
};
// Convert string to MissionPhase
inline MissionPhase stringToMissionPhase(const std::string& phaseStr) {
    if (phaseStr == "LAUNCH") return MissionPhase::LAUNCH;
    if (phaseStr == "TRANSIT") return MissionPhase::TRANSIT;
    if (phaseStr == "EXPLORATION") return MissionPhase::EXPLORATION;
    if (phaseStr == "RETURN") return MissionPhase::RETURN;
    if (phaseStr == "LANDING") return MissionPhase::LANDING;
    if (phaseStr == "EMERGENCY") return MissionPhase::EMERGENCY;
    return MissionPhase::LAUNCH; // Default
}
// Convert CommandPriority to string
inline std::string priorityToString(CommandPriority priority) {
    switch (priority) {
        case CommandPriority::EMERGENCY: return "EMERGENCY";
        case CommandPriority::HIGH: return "HIGH";
        case CommandPriority::NORMAL: return "NORMAL";
        case CommandPriority::LOW: return "LOW";
        default: return "NORMAL";
    }
}

// Convert string to CommandPriority
inline CommandPriority stringToPriority(const std::string& priorityStr) {
    if (priorityStr == "EMERGENCY") return CommandPriority::EMERGENCY;
    if (priorityStr == "HIGH") return CommandPriority::HIGH;
    if (priorityStr == "NORMAL") return CommandPriority::NORMAL;
    if (priorityStr == "LOW") return CommandPriority::LOW;
    return CommandPriority::NORMAL; // Default
}
// Convert MissionPhase to string
inline std::string missionPhaseToString(MissionPhase phase) {
    switch (phase) {
        case MissionPhase::LAUNCH: return "LAUNCH";
        case MissionPhase::TRANSIT: return "TRANSIT";
        case MissionPhase::EXPLORATION: return "EXPLORATION";
        case MissionPhase::RETURN: return "RETURN";
        case MissionPhase::LANDING: return "LANDING";
        case MissionPhase::EMERGENCY: return "EMERGENCY";
        default: return "UNKNOWN";
    }
}





#endif //PACKAGEM_H
