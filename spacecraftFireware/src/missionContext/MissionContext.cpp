//
// Created by DevAccount on 04/04/2025.
//

#include <missionContext/MissionContext.h>

json MissionContext::toJson() const {
    return {
                {"current_step", currentStep},
                {"total_steps", totalSteps},
                {"mission_objectives", missionObjectives},
                {"command_cooldowns", commandCooldowns},
                {"mission_parameters", missionParameters},
                {"execution_history", executionHistory},
                {"current_mission_id", currentMissionId}
    };
}
MissionContext MissionContext::fromJson(const json& data) {
    MissionContext context;

    if (data.contains("current_step")) context.currentStep = data["current_step"].get<int>();
    if (data.contains("total_steps")) context.totalSteps = data["total_steps"].get<int>();
    if (data.contains("mission_objectives")) {
        for (auto& [key, value] : data["mission_objectives"].items()) {
            context.missionObjectives[key] = value;
        }
    }
    if (data.contains("command_cooldowns")) {
        for (auto& [key, value] : data["command_cooldowns"].items()) {
            context.commandCooldowns[key] = value.get<double>();
        }
    }
    if (data.contains("mission_parameters")) context.missionParameters = data["mission_parameters"];
    if (data.contains("execution_history")) {
        context.executionHistory = data["execution_history"].get<std::vector<std::string>>();
    }
    if (data.contains("current_mission_id")) {
        context.currentMissionId = data["current_mission_id"].get<std::string>();
    }

    return context;
}
