//
// Created by DevAccount on 04/04/2025.
//

#include <missionContext/SpacecraftStatus.h>



SpacecraftStatus::SpacecraftStatus()
        : fuelLevel(100.0), position{0.0, 0.0, 0.0}, velocity{0.0, 0.0, 0.0},
          missionPhase(MissionPhase::LAUNCH) {
    lastUpdateTime = getCurrentTime();
}

json SpacecraftStatus::toJson() const {
    return {
                {"fuel_level", fuelLevel},
                {"position", position},
                {"velocity", velocity},
                {"anomaly_flags", anomalyFlags},
                {"mission_phase", missionPhaseToString(missionPhase)},
                {"environmental_data", environmentalData},
                {"system_health", systemHealth},
                {"last_update_time", lastUpdateTime}
    };
}

SpacecraftStatus SpacecraftStatus::fromJson(const json &data) {
    SpacecraftStatus status;

    if (data.contains("fuel_level")) status.fuelLevel = data["fuel_level"].get<double>();
    if (data.contains("position")) status.position = data["position"].get<std::vector<double>>();
    if (data.contains("velocity")) status.velocity = data["velocity"].get<std::vector<double>>();
    if (data.contains("anomaly_flags")) {
        for (auto& [key, value] : data["anomaly_flags"].items()) {
            status.anomalyFlags[key] = value.get<bool>();
        }
    }
    if (data.contains("mission_phase")) {
        status.missionPhase = stringToMissionPhase(data["mission_phase"].get<std::string>());
    }
    if (data.contains("environmental_data")) status.environmentalData = data["environmental_data"];
    if (data.contains("system_health")) status.systemHealth = data["system_health"];
    if (data.contains("last_update_time")) status.lastUpdateTime = data["last_update_time"].get<double>();

    return status;
}

