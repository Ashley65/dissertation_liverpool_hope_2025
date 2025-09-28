//
// Created by DevAccount on 04/04/2025.
//

#ifndef MISSIONCONTEXT_H
#define MISSIONCONTEXT_H
#pragma once

#include <packagem.h>



class MissionContext {
    public:
        int currentStep;
        int totalSteps;
        std::map<std::string, json> missionObjectives;
        std::map<std::string, double> commandCooldowns;
        json missionParameters;
        std::vector<std::string> executionHistory;
        std::string currentMissionId;

    MissionContext() : currentStep(0), totalSteps(0) {}
    json toJson() const;
    static MissionContext fromJson(const json& data);


};



#endif //MISSIONCONTEXT_H
