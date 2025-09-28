//
// Created by DevAccount on 04/04/2025.
//

#include <commad/command.h>


Command::Command(const std::string &type, const json &params, CommandPriority prio) : commandType(type),
    parameters(params), priority(prio) {
    timestamp = getCurrentTime();
    executionId = type + "-" + std::to_string(timestamp);
    status = "pending";
    result = nullptr;
}

json Command::toJson() const {
    return {
                {"command_type", commandType},
                {"parameters", parameters},
                {"priority", priorityToString(priority)},
                {"timestamp", timestamp},
                {"execution_id", executionId},
                {"status", status},
                {"result", result}
    };
}

Command Command::fromJson(const json &data) {
    Command cmd(
            data["command_type"].get<std::string>(),
            data.contains("parameters") ? data["parameters"] : json({}),
            data.contains("priority") ? stringToPriority(data["priority"].get<std::string>()) : CommandPriority::NORMAL
        );

    if (data.contains("timestamp")) cmd.timestamp = data["timestamp"].get<double>();
    if (data.contains("execution_id")) cmd.executionId = data["execution_id"].get<std::string>();
    if (data.contains("status")) cmd.status = data["status"].get<std::string>();
    if (data.contains("result")) cmd.result = data["result"];

    return cmd;
}
