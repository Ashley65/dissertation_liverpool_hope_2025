//
// Created by DevAccount on 03/04/2025.
//

#ifndef SPACECRAFTSERVER_H
#define SPACECRAFTSERVER_H

#include <packagem.h>
#include <commad/command.h>
#include <missionContext/MissionContext.h>
#include <missionContext/SpacecraftStatus.h>
#include <safety/safetyManager.h>




// Command comparison function for priority queue
struct CommandComparator {
    bool operator()(const Command& a, const Command& b) {
        // Lower value = higher priority
        return static_cast<int>(a.priority) > static_cast<int>(b.priority);
    }
};




class spacecraftServer {
public:


    spacecraftServer(const std::string &address);

    ~spacecraftServer();

    void start();
    // Method for testing - sends a command and waits for response
    std::string sendMessage(const std::string& message) {
        if (!running || !socket) {
            throw std::runtime_error("Server not running");
        }

        // Create message
        zmq::message_t request(message.size());
        memcpy(request.data(), message.c_str(), message.size());

        // Send the message
        socket->send(request, zmq::send_flags::none);

        // Wait for reply
        zmq::message_t reply;
        auto result = socket->recv(reply, zmq::recv_flags::none);

        // Return reply as string
        return std::string(static_cast<char*>(reply.data()), reply.size());
    }
    void stop();
private:
    std::string address;
    std::unique_ptr<zmq::context_t> context;
    std::unique_ptr<zmq::socket_t> socket;
    std::atomic<bool> running;
    std::thread statusThread;

    enum class ConnectionType {
        AI_SYSTEM,
        ENVIRONMENT,
        UNKNOWN
    };

    // Socket for environment connections
    std::unique_ptr<zmq::socket_t> environmentSocket;

    // Environment processing thread
    std::thread environmentThread;

    // Connection tracking
    std::map<std::string, ConnectionType> connectedClients;
    std::mutex clientsMutex;

    // Methods to handle different connection types
    void environmentMessageLoop();
    json processEnvironmentMessage(const std::string& messageStr);
    void authenticateConnection(const std::string& clientId, const json& authData);

    SpacecraftStatus spacecraftStatus;
    MissionContext missionContext;

    std::priority_queue<Command, std::vector<Command>, CommandComparator> commandQueue;
    std::vector<Command> commandHistory;

    std::map<std::string, std::function<json(const json&)>> commandHandlers;
    std::map<std::string, double> commandCooldowns;

    std::mutex statusMutex;
    std::mutex queueMutex;
    std::unique_ptr<SafetyManager> safetyManager;
    void checkSafetyStatus();
    void initialiseCommandHandlers();
    void initialiseCommandCooldowns();
    void messageLoop();
    void reconnect();
    json processMessage(const std::string& messageStr);
    json handleStatusRequest();
    json handleCommandMessage(const json& data);
    json handleMissionUpdate(const json& data);
    bool isCommandOnCooldown(const std::string& commandType);
    void setCommandCooldown(const std::string& commandType);
    json executeCommand(Command& command);
    void updateStatus();
    void updateSpacecraftStatus();
    void processCommandQueue();
    json handleEmergencyProtocol(const json& parameters);
    json handleInvestigateAnomaly(const json& parameters);
    json handleAdjustTrajectory(const json& parameters);
    json handleIncreaseVelocity(const json& parameters);
    json handleDecreaseVelocity(const json& parameters);
    json handleMaintainCourse(const json& parameters);
    double getCurrentTime() {
        return duration_cast<milliseconds>(
            system_clock::now().time_since_epoch()
        ).count() / 1000.0;
    }





};



#endif //SPACECRAFTSERVER_H
