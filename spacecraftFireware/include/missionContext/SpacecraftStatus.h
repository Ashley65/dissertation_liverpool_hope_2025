//
// Created by DevAccount on 04/04/2025.
//

#ifndef SPACECRAFTSTATUS_H
#define SPACECRAFTSTATUS_H
#pragma once
#include <packagem.h>


class SpacecraftStatus {
    public:
        double fuelLevel;
        std::vector<double> position;
        std::vector<double> velocity;
        std::map<std::string, bool> anomalyFlags;
        MissionPhase missionPhase;
        json environmentalData;
        json systemHealth;
        double lastUpdateTime;

        SpacecraftStatus();
        json toJson() const;
        static SpacecraftStatus fromJson(const json& data);
    private:
        static double getCurrentTime() {
            return duration_cast<milliseconds>(
            system_clock::now().time_since_epoch()).count() / 1000.0;
        }

};



#endif //SPACECRAFTSTATUS_H
