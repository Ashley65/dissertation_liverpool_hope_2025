//
// Created by DevAccount on 03/04/2025.
//

#include <networking/spacecraftServer.h>
#include "logger/LogMacros.h"


spacecraftServer::spacecraftServer(const std::string &address) :
    address(address),
    running(false),
    safetyManager(std::make_unique<SafetyManager>()) {

    // Only store the address and parse environment address
    std::string envAddress = address;
    size_t colonPos = envAddress.find_last_of(':');
    if (colonPos != std::string::npos) {
        int port = std::stoi(envAddress.substr(colonPos + 1));
        envAddress = envAddress.substr(0, colonPos + 1) + std::to_string(port + 1);
    }

    // Initialize command handlers and cooldowns
    initialiseCommandHandlers();
    initialiseCommandCooldowns();
}

spacecraftServer::~spacecraftServer() {
    stop();
}
void spacecraftServer::start() {
    if (running) {
        LOG_WARNING("Server is already running");
        return;
    }

    try {
        LOG_INFO("Binding to " + address);
        context = std::make_unique<zmq::context_t>(1);
        socket = std::make_unique<zmq::socket_t>(*context, ZMQ_REP);
        socket->bind(address);
        LOG_INFO("Server bound to " + address);

        // Create and bind the environment socket
        std::string envAddress = address;
        size_t colonPos = envAddress.find_last_of(':');
        if (colonPos != std::string::npos) {
            int port = std::stoi(envAddress.substr(colonPos + 1));
            envAddress = envAddress.substr(0, colonPos + 1) + std::to_string(port + 1);
        }

        environmentSocket = std::make_unique<zmq::socket_t>(*context, ZMQ_REP);
        environmentSocket->bind(envAddress);
        LOG_INFO("Environment socket bound to " + envAddress);

        safetyManager->start();
        running = true;
        LOG_INFO("Server started successfully");

        // Start status update thread
        statusThread = std::thread(&spacecraftServer::updateStatus, this);

        // Start environment message thread
        environmentThread = std::thread(&spacecraftServer::environmentMessageLoop, this);

        // Start message loop
        messageLoop();
    }
    catch (const std::exception& e) {
        LOG_ERROR("Failed to start server: " + std::string(e.what()));
        throw;
    }
}

void spacecraftServer::stop() {
    if (!running) return;

    LOG_INFO("Stopping server...");
    running = false;

    if (statusThread.joinable()) {
        statusThread.join();
    }
    if (environmentThread.joinable()) {
        environmentThread.join();
    }
    if (safetyManager) {
        safetyManager->stop();
        safetyManager.reset();
    }

    if (socket) {
        socket->close();
        socket.reset();
    }
    if (environmentSocket) {
        environmentSocket->close();
        environmentSocket.reset();
    }

    if (context) {
        context->close();
        context.reset();
    }

    LOG_INFO("Server stopped");
}

void spacecraftServer::environmentMessageLoop() {
    LOG_INFO("Starting environment message loop");

    while (running.load()) {
        try {
            zmq::message_t message;

            // Use poll with timeout to allow checking running state
            zmq::pollitem_t items[] = {
                { static_cast<void*>(*environmentSocket), 0, ZMQ_POLLIN, 0 }
            };

            zmq::poll(&items[0], 1, std::chrono::milliseconds(500));

            if (items[0].revents & ZMQ_POLLIN) {
                auto result = environmentSocket->recv(message, zmq::recv_flags::none);

                if (result) {
                    std::string messageStr(static_cast<char*>(message.data()), message.size());
                    LOG_DEBUG("Environment message received: " + messageStr);

                    // Try to parse the message
                    try {
                        json response = processEnvironmentMessage(messageStr);
                        std::string responseStr = response.dump();

                        LOG_DEBUG("Sending environment response: " + responseStr);
                        zmq::message_t reply(responseStr.size());
                        memcpy(reply.data(), responseStr.c_str(), responseStr.size());
                        environmentSocket->send(reply, zmq::send_flags::none);
                    } catch (const json::exception& e) {
                        LOG_ERROR("Error parsing environment message: " + std::string(e.what()));
                        // Send error response
                        json errorResponse = {
                            {"status", "error"},
                            {"message", "Invalid message format"}
                        };
                        std::string responseStr = errorResponse.dump();
                        zmq::message_t reply(responseStr.size());
                        memcpy(reply.data(), responseStr.c_str(), responseStr.size());
                        environmentSocket->send(reply, zmq::send_flags::none);
                    }
                }
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Error in environment message loop: " + std::string(e.what()));
            // Sleep before attempting to continue
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    LOG_INFO("Environment message loop stopped");
}

json spacecraftServer::processEnvironmentMessage(const std::string& messageStr) {
    try {
        json message = json::parse(messageStr);

        if (!message.contains("type")) {
            return {{"status", "error"}, {"message", "Message type not specified"}};
        }

        std::string messageType = message["type"];

        if (messageType == "authenticate") {
            if (!message.contains("clientId") || !message.contains("authData")) {
                return {{"status", "error"}, {"message", "Missing authentication data"}};
            }

            std::string clientId = message["clientId"];
            authenticateConnection(clientId, message["authData"]);

            return {{"status", "success"}, {"message", "Authentication successful"}};
        }
        else if (messageType == "environmentUpdate") {
            if (!message.contains("data")) {
                return {{"status", "error"}, {"message", "Missing environment data"}};
            }

            // Update spacecraft status with environment data
            std::lock_guard<std::mutex> lock(statusMutex);
            spacecraftStatus.environmentalData = message["data"];
            spacecraftStatus.lastUpdateTime = getCurrentTime();

            // Check if this update triggers any safety concerns
            safetyManager->evaluateEnvironmentalData(spacecraftStatus.environmentalData);
            checkSafetyStatus();

            return {{"status", "success"}};
        }
        else {
            return {{"status", "error"}, {"message", "Unknown message type"}};
        }
    }
    catch (const std::exception& e) {
        LOG_ERROR("Error processing environment message: " + std::string(e.what()));
        return {{"status", "error"}, {"message", std::string(e.what())}};
    }
}

void spacecraftServer::authenticateConnection(const std::string& clientId, const json& authData) {
    // In a real system, I would validate credentials here
    // For this example, we'll just check the connection type

    std::lock_guard<std::mutex> lock(clientsMutex);

    if (authData.contains("type")) {
        std::string type = authData["type"];
        if (type == "ai_system") {
            connectedClients[clientId] = ConnectionType::AI_SYSTEM;
            LOG_INFO("AI system connected: " + clientId);
        }
        else if (type == "environment") {
            connectedClients[clientId] = ConnectionType::ENVIRONMENT;
            LOG_INFO("Environment system connected: " + clientId);
        }
        else {
            connectedClients[clientId] = ConnectionType::UNKNOWN;
            LOG_WARNING("Unknown connection type for client: " + clientId);
        }
    }
    else {
        connectedClients[clientId] = ConnectionType::UNKNOWN;
        LOG_WARNING("Client connected without type specification: " + clientId);
    }
}

void spacecraftServer::checkSafetyStatus() {
    if (!safetyManager->isSystemHealthy()) {
        LOG_WARNING("System health check failed");
        if (spacecraftStatus.fuelLevel < 10.0) {
            safetyManager->triggerEmergencyProtocol("Critical fuel level");
        }
        if (spacecraftStatus.missionPhase == MissionPhase::EMERGENCY) {
            safetyManager->triggerEmergencyProtocol("Mission phase is EMERGENCY");
        }
        if (spacecraftStatus.anomalyFlags["critical_system_failure"]) {
            safetyManager->triggerEmergencyProtocol("Critical system failure detected");
        }
        if (spacecraftStatus.anomalyFlags["environmental_anomaly"]) {
            safetyManager->triggerEmergencyProtocol("Environmental anomaly detected");
        }
        if (spacecraftStatus.anomalyFlags["system_overload"]) {
            safetyManager->triggerEmergencyProtocol("System overload detected");
        }
        if (spacecraftStatus.anomalyFlags["communication_failure"]) {
            safetyManager->triggerEmergencyProtocol("Communication failure detected");
        }
        if (spacecraftStatus.anomalyFlags["navigation_error"]) {
            safetyManager->triggerEmergencyProtocol("Navigation error detected");
        }
        if (spacecraftStatus.anomalyFlags["power_failure"]) {
            safetyManager->triggerEmergencyProtocol("Power failure detected");
        }
        if (spacecraftStatus.anomalyFlags["software_error"]) {
            safetyManager->triggerEmergencyProtocol("Software error detected");
        }

    }

}

void spacecraftServer::initialiseCommandHandlers() {
    commandHandlers["emergency_protocol"] = [this](const json& params) {
        return handleEmergencyProtocol(params);
    };
    commandHandlers["investigate_anomaly"] = [this](const json& params) {
        return handleInvestigateAnomaly(params);
    };
    commandHandlers["adjust_trajectory"] = [this](const json& params) {
        return handleAdjustTrajectory(params);
    };
    commandHandlers["increase_velocity"] = [this](const json& params) {
        return handleIncreaseVelocity(params);
    };
    commandHandlers["decrease_velocity"] = [this](const json& params) {
        return handleDecreaseVelocity(params);
    };
    commandHandlers["maintain_course"] = [this](const json& params) {
        return handleMaintainCourse(params);
    };
}

void spacecraftServer::initialiseCommandCooldowns() {
    commandCooldowns["emergency_protocol"] = 0.0;  // No cooldown for emergency
    commandCooldowns["investigate_anomaly"] = 10.0;
    commandCooldowns["adjust_trajectory"] = 5.0;
    commandCooldowns["increase_velocity"] = 3.0;
    commandCooldowns["decrease_velocity"] = 3.0;
    commandCooldowns["maintain_course"] = 2.0;
}

void spacecraftServer::messageLoop() {
    zmq::pollitem_t items[] = {
        { static_cast<void*>(*socket), 0, ZMQ_POLLIN, 0 }
    };

    while (running) {
        try {
            // Poll with timeout (1000 ms = 1 sec)
            zmq::poll(items, 1, 1000);

            if (items[0].revents & ZMQ_POLLIN) {
                // Receive a message
                zmq::message_t message;
                socket->recv(message);
                std::string messageStr(static_cast<char*>(message.data()), message.size());
                LOG_DEBUG("Received message: " + messageStr);

                // Process message
                json response = processMessage(messageStr);
                std::string responseStr = response.dump();

                // Send response
                zmq::message_t reply(responseStr.size());
                memcpy(reply.data(), responseStr.c_str(), responseStr.size());
                socket->send(reply);
                LOG_DEBUG("Sent response: " + responseStr);
            }
        }
        catch (const zmq::error_t& e) {
            if (running) {
                LOG_ERROR("ZMQ Error: " + std::string(e.what()));
                reconnect();
            }
        }
        catch (const std::exception& e) {
            LOG_ERROR("Error in message loop: " + std::string(e.what()));
        }
    }
}

void spacecraftServer::reconnect() {
    try {
        if (socket) {
            socket->close();
            socket.reset();
        }

        socket = std::make_unique<zmq::socket_t>(*context, ZMQ_REP);
        socket->bind(address);
        LOG_INFO("Successfully reconnected");
    }
    catch (const std::exception& e) {
        LOG_ERROR("Failed to reconnect: " + std::string(e.what()));
        std::this_thread::sleep_for(std::chrono::seconds(5)); // Wait before retrying
    }
}

json spacecraftServer::processMessage(const std::string &messageStr) {
    try {
        json data = json::parse(messageStr);
        std::string messageType = data.contains("type") ? data["type"].get<std::string>() : "";

        if (messageType == "status_request") {
            return handleStatusRequest();
        }
        else if (messageType == "command") {
            return handleCommandMessage(data);
        }
        else if (messageType == "mission_update") {
            return handleMissionUpdate(data);
        }
        else {
            LOG_WARNING("Unknown message type: " + messageType);
            return {
                        {"status", "error"},
                        {"message", "Unknown message type: " + messageType}
            };
        }
    }
    catch (const json::parse_error& e) {
        LOG_ERROR("Failed to decode JSON message: " + std::string(e.what()));
        return {
                    {"status", "error"},
                    {"message", "Invalid JSON format"}
        };
    }
    catch (const std::exception& e) {
        LOG_ERROR("Error processing message: " + std::string(e.what()));
        return {
                    {"status", "error"},
                    {"message", std::string(e.what())}
        };
    }
}

json spacecraftServer::handleStatusRequest() {
    std::lock_guard<std::mutex> lock(statusMutex);

    return {
                {"status", "success"},
                {"data", {
                    {"spacecraft_status", spacecraftStatus.toJson()},
                    {"mission_context", missionContext.toJson()}
                }}
    };
}

json spacecraftServer::handleCommandMessage(const json &data) {
    if (!data.contains("command")) {
        return {
                    {"status", "error"},
                    {"message", "No command provided"}
        };
    }

    json commandData = data["command"];
    if (!commandData.contains("command_type")) {
        return {
                    {"status", "error"},
                    {"message", "No command type specified"}
        };
    }

    std::string commandType = commandData["command_type"].get<std::string>();

    // Check if command is supported
    if (commandHandlers.find(commandType) == commandHandlers.end()) {
        return {
                    {"status", "error"},
                    {"message", "Unsupported command: " + commandType}
        };
    }

    // Check for command cooldown
    if (isCommandOnCooldown(commandType)) {
        return {
                    {"status", "cooldown"},
                    {"message", "Command " + commandType + " is on cooldown"}
        };
    }

    // Create command object
    Command command(
        commandType,
        commandData.contains("parameters") ? commandData["parameters"] : json({}),
        commandData.contains("priority") ?
            stringToPriority(commandData["priority"].get<std::string>()) :
            CommandPriority::NORMAL
    );

    // For emergency commands, execute immediately
    if (command.priority == CommandPriority::EMERGENCY) {
        return executeCommand(command);
    }

    // Otherwise add to queue
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        commandQueue.push(command);
    }

    return {
                {"status", "queued"},
                {"execution_id", command.executionId},
                {"message", "Command " + commandType + " queued successfully"}
    };
}

json spacecraftServer::handleMissionUpdate(const json &data) {
    if (!data.contains("mission_data")) {
        return {
                    {"status", "error"},
                    {"message", "No mission data provided"}
        };
    }

    json missionData = data["mission_data"];

    // Update mission context
    {
        std::lock_guard<std::mutex> lock(statusMutex);
        missionContext = MissionContext::fromJson(missionData);
    }

    return {
                {"status", "success"},
                {"message", "Mission context updated"}
    };
}

bool spacecraftServer::isCommandOnCooldown(const std::string &commandType) {
    if (commandCooldowns.find(commandType) == commandCooldowns.end()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(statusMutex);
    auto it = missionContext.commandCooldowns.find(commandType);
    if (it == missionContext.commandCooldowns.end()) {
        return false;
    }

    return it->second > getCurrentTime();
}

void spacecraftServer::setCommandCooldown(const std::string &commandType) {
    auto it = commandCooldowns.find(commandType);
    if (it == commandCooldowns.end()) {
        return;
    }

    double cooldownDuration = it->second;
    double cooldownEndTime = getCurrentTime() + cooldownDuration;

    std::lock_guard<std::mutex> lock(statusMutex);
    missionContext.commandCooldowns[commandType] = cooldownEndTime;
}

json spacecraftServer::executeCommand(Command &command) {
    LOG_INFO("Executing command: " + command.commandType);

    // Get the appropriate handler
    auto handlerIt = commandHandlers.find(command.commandType);
    if (handlerIt == commandHandlers.end()) {
        command.status = "failed";
        command.result = {{"error", "No handler for command type: " + command.commandType}};
        return {
                    {"status", "error"},
                    {"message", command.result["error"]},
                    {"execution_id", command.executionId}
        };
    }

    try {
        // Execute the command
        json result = handlerIt->second(command.parameters);
        command.status = "completed";
        command.result = result;

        // Set command cooldown
        setCommandCooldown(command.commandType);

        // Add to command history
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            commandHistory.push_back(command);
            // Keep history manageable
            if (commandHistory.size() > 100) {
                commandHistory.erase(commandHistory.begin(), commandHistory.begin() +
                                   (commandHistory.size() - 100));
            }
        }

        // Add to execution history in mission context
        {
            std::lock_guard<std::mutex> lock(statusMutex);
            missionContext.executionHistory.push_back(command.executionId);
        }

        return {
                    {"status", "success"},
                    {"execution_id", command.executionId},
                    {"result", result}
        };
    }
    catch (const std::exception& e) {
        LOG_ERROR("Error executing command " + command.commandType + ": " + std::string(e.what()));
        command.status = "failed";
        command.result = {{"error", std::string(e.what())}};
        return {
                    {"status", "error"},
                    {"execution_id", command.executionId},
                    {"message", std::string(e.what())}
        };
    }
}

void spacecraftServer::updateStatus() {
    while (running) {
        try {
            // Update status from simulation
            updateSpacecraftStatus();

            // Process command queue
            processCommandQueue();

            std::this_thread::sleep_for(std::chrono::seconds(1)); // Update every second
        }
        catch (const std::exception& e) {
            LOG_ERROR("Error in status update: " + std::string(e.what()));
        }
    }
}

void spacecraftServer::updateSpacecraftStatus() {
    // In a real implementation, this would get data from sensors or simulation
    std::lock_guard<std::mutex> lock(statusMutex);
    spacecraftStatus.lastUpdateTime = getCurrentTime();

    // Simulate fuel consumption
    if (spacecraftStatus.fuelLevel > 0) {
        spacecraftStatus.fuelLevel -= 0.01;  // Small consumption rate
    }
}

void spacecraftServer::processCommandQueue() {
    Command command(""); // Default initialization
    bool hasCommand = false;

    {
        std::lock_guard<std::mutex> lock(queueMutex);
        if (!commandQueue.empty()) {
            command = commandQueue.top();
            commandQueue.pop();
            hasCommand = true;
        }
    }

    if (hasCommand) {
        executeCommand(command);
    }
}

json spacecraftServer::handleEmergencyProtocol(const json &parameters) {
    std::string emergencyType = parameters.contains("emergency_type") ?
                                   parameters["emergency_type"].get<std::string>() : "general";
    LOG_WARNING("EMERGENCY PROTOCOL ACTIVATED: " + emergencyType);

    {
        std::lock_guard<std::mutex> lock(statusMutex);
        // Set mission phase to EMERGENCY
        spacecraftStatus.missionPhase = MissionPhase::EMERGENCY;
        // Add emergency flag
        spacecraftStatus.anomalyFlags["emergency_" + emergencyType] = true;
    }

    return {
                {"message", "Emergency protocol activated for " + emergencyType},
                {"status_change", "mission_phase=EMERGENCY"}
    };

}

json spacecraftServer::handleInvestigateAnomaly(const json &parameters) {
    std::string anomalyType = parameters.contains("anomaly_type") ?
                                 parameters["anomaly_type"].get<std::string>() : "unknown";
    std::string location = parameters.contains("location") ?
                          parameters["location"].get<std::string>() : "unspecified";
    LOG_INFO("Investigating anomaly: " + anomalyType + " at " + location);

    // Simulate investigation
    json investigationResult = {
        {"anomaly_confirmed", true},
        {"severity", "medium"},
        {"description", "Confirmed " + anomalyType + " anomaly at " + location},
        {"recommended_action", "adjust_trajectory"}
    };

    return {
                {"message", "Investigation complete for " + anomalyType + " anomaly"},
                {"findings", investigationResult}
    };
}

json spacecraftServer::handleAdjustTrajectory(const json &parameters) {
    std::vector<double> vector = parameters.contains("vector") ?
                                    parameters["vector"].get<std::vector<double>>() :
                                    std::vector<double>{0, 0, 0};
    double magnitude = parameters.contains("magnitude") ?
                      parameters["magnitude"].get<double>() : 1.0;

    LOG_INFO("Adjusting trajectory: magnitude=" + std::to_string(magnitude));

    {
        std::lock_guard<std::mutex> lock(statusMutex);
        // Apply the trajectory adjustment to velocity
        for (size_t i = 0; i < 3 && i < vector.size(); ++i) {
            spacecraftStatus.velocity[i] += vector[i] * magnitude;
        }
    }

    return {
                {"message", "Trajectory adjusted successfully"},
                {"new_velocity", spacecraftStatus.velocity}
    };
}

json spacecraftServer::handleIncreaseVelocity(const json &parameters) {
    double amount = parameters.contains("amount") ?
                       parameters["amount"].get<double>() : 1.0;
    LOG_INFO("Increasing velocity by " + std::to_string(amount));

    std::vector<double> newVelocity;
    double newFuelLevel;

    {
        std::lock_guard<std::mutex> lock(statusMutex);
        // Calculate current speed
        double currentSpeed = 0;
        for (double v : spacecraftStatus.velocity) {
            currentSpeed += v * v;
        }
        currentSpeed = std::sqrt(currentSpeed);

        if (currentSpeed == 0) {
            // If not moving, set a default direction
            spacecraftStatus.velocity = {amount, 0, 0};
        }
        else {
            // Scale velocity vector
            double scaleFactor = (currentSpeed + amount) / currentSpeed;
            for (size_t i = 0; i < spacecraftStatus.velocity.size(); ++i) {
                spacecraftStatus.velocity[i] *= scaleFactor;
            }
        }

        // Simulate fuel consumption
        spacecraftStatus.fuelLevel -= amount * 0.1;

        newVelocity = spacecraftStatus.velocity;
        newFuelLevel = spacecraftStatus.fuelLevel;
    }

    return {
                {"message", "Velocity increased by " + std::to_string(amount)},
                {"new_velocity", newVelocity},
                {"fuel_level", newFuelLevel}
    };
}

json spacecraftServer::handleDecreaseVelocity(const json &parameters) {
    double amount = parameters.contains("amount") ?
                       parameters["amount"].get<double>() : 1.0;
    LOG_INFO("Decreasing velocity by " + std::to_string(amount));

    std::vector<double> newVelocity;
    double newSpeed;
    double newFuelLevel;

    {
        std::lock_guard<std::mutex> lock(statusMutex);
        // Calculate current speed
        double currentSpeed = 0;
        for (double v : spacecraftStatus.velocity) {
            currentSpeed += v * v;
        }
        currentSpeed = std::sqrt(currentSpeed);

        if (currentSpeed <= amount) {
            // Full stop
            spacecraftStatus.velocity = {0, 0, 0};
            newSpeed = 0;
        }
        else {
            // Scale velocity vector
            double scaleFactor = (currentSpeed - amount) / currentSpeed;
            for (size_t i = 0; i < spacecraftStatus.velocity.size(); ++i) {
                spacecraftStatus.velocity[i] *= scaleFactor;
            }
            newSpeed = currentSpeed - amount;
        }

        // Simulate fuel consumption
        spacecraftStatus.fuelLevel -= amount * 0.05;

        newVelocity = spacecraftStatus.velocity;
        newFuelLevel = spacecraftStatus.fuelLevel;
    }

    return {
                {"message", "Velocity decreased by " + std::to_string(amount)},
                {"new_velocity", newVelocity},
                {"new_speed", newSpeed},
                {"fuel_level", newFuelLevel}
    };
}

json spacecraftServer::handleMaintainCourse(const json &parameters) {
    int duration = parameters.contains("duration") ?
                      parameters["duration"].get<int>() : 60;  // seconds
    LOG_INFO("Maintaining current course for " + std::to_string(duration) + " seconds");

    std::vector<double> currentVelocity;
    {
        std::lock_guard<std::mutex> lock(statusMutex);
        currentVelocity = spacecraftStatus.velocity;
    }

    return {
                {"message", "Maintaining course for " + std::to_string(duration) + " seconds"},
                {"current_velocity", currentVelocity}
    };
}
















