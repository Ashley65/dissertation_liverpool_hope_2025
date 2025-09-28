//
// Created by DevAccount on 04/04/2025.
//

#ifndef COMMAND_H
#define COMMAND_H
#pragma once
#include <packagem.h>


class Command {
    public:
        std::string commandType;
        json parameters;
        CommandPriority priority;
        double timestamp;
        std::string executionId;
        std::string status;
        json result;

        explicit Command(const std::string& type, const json& params = json(),
            CommandPriority prio = CommandPriority::NORMAL);
        json toJson() const;
        static Command fromJson(const json& data);
    private:
        static double getCurrentTime() {
            return duration_cast<milliseconds>(
            system_clock::now().time_since_epoch()).count() / 1000.0;
        }
};







#endif //COMMAND_H
