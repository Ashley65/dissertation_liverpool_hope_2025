
#ifndef FIRMWARE_SERVER_CUH
#define FIRMWARE_SERVER_CUH
#pragma once
#include <iostream>
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <physics/Body.cuh>

#include <ai/AIDataPacket.cuh>

class firmwareClient {
private:
    zmq::context_t context;
    zmq::socket_t socket;
    std::string endpoint;
    bool connected;

public:
    firmwareClient(const std::string& endpoint = "tcp://localhost:5555")
        : context(1), socket(context, zmq::socket_type::req), endpoint(endpoint), connected(false) {
        try {
            socket.connect(endpoint);
            connected = true;
        } catch (const zmq::error_t& e) {
            std::cerr << "Error connecting to firmware server: " << e.what() << std::endl;
            connected = false;
        }
    }

    ~firmwareClient() {
        if (connected) {
            socket.disconnect(endpoint);
        }
    }

    bool sendSimuationState(const std::vector<Body>& bodies, const std::vector<spaceship>& ships);

    // New method to send captain data packet to the AI server
    bool sendCaptainData(const CaptainDataPacket& dataPacket) {
        if (!connected) {
            std::cerr << "Not connected to firmware server." << std::endl;
            return false;
        }

        nlohmann::json captainData;
        captainData["type"] = "captain_data";

        // Position data
        captainData["position"] = {
            dataPacket.getPositionX(),
            dataPacket.getPositionY(),
            dataPacket.getPositionZ()
        };
        captainData["waypoint_distance"] = dataPacket.getWaypointDistance();
        captainData["target_distance"] = dataPacket.getTargetDistance();

        // Spacecraft state
        captainData["velocity"] = {
            dataPacket.getVelocityX(),
            dataPacket.getVelocityY(),
            dataPacket.getVelocityZ()
        };
        captainData["speed"] = dataPacket.getSpeed();
        captainData["fuel"] = dataPacket.getFuel();
        captainData["mission_time"] = dataPacket.getMissionTime();

        // Mission context
        captainData["mission_phase"] = static_cast<int>(dataPacket.getPhase());
        captainData["mission_progress"] = dataPacket.getMissionProgress();

        // Waypoints
        captainData["waypoints"] = nlohmann::json::array();
        for (const auto& waypoint : dataPacket.getWaypoints()) {
            nlohmann::json wp;
            wp["position"] = {waypoint.x, waypoint.y, waypoint.z};
            wp["name"] = waypoint.name;
            wp["reached"] = waypoint.reached;
            captainData["waypoints"].push_back(wp);
        }

        // Environmental data
        const auto& env = dataPacket.getEnvironment();
        captainData["environment"] = {
            {"temperature", env.temperature},
            {"radiation", env.radiation},
            {"magnetic_field", env.magneticField},
            {"hazards", env.hazards}
        };

        // Anomaly information
        const auto& anomaly = dataPacket.getAnomaly();
        captainData["anomaly"] = {
            {"present", anomaly.present},
            {"type", anomaly.type},
            {"severity", anomaly.severity},
            {"confidence", anomaly.confidence}
        };

        // Memory records
        captainData["memories"] = nlohmann::json::array();
        for (const auto& memory : dataPacket.getMemories()) {
            nlohmann::json mem;
            mem["context"] = memory.context;
            mem["relevance"] = memory.relevance;
            mem["experience"] = memory.experience;
            mem["utilization"] = memory.utilization;
            captainData["memories"].push_back(mem);
        }

        // Performance metrics
        const auto& perf = dataPacket.getPerformance();
        captainData["performance"] = {
            {"reward", perf.reward},
            {"path_efficiency", perf.pathEfficiency},
            {"fuel_efficiency", perf.fuelEfficiency},
            {"anomaly_handling", perf.anomalyHandlingScore}
        };

        // Prepare and send the message
        std::string captainData_str = captainData.dump();
        zmq::message_t message(captainData_str.size());
        memcpy(message.data(), captainData_str.data(), captainData_str.size());

        auto send_result = socket.send(message, zmq::send_flags::none);
        if (!send_result) {
            std::cerr << "Error sending captain data to firmware server." << std::endl;
            return false;
        }

        // Receive response
        zmq::message_t reply;
        auto recv_result = socket.recv(reply, zmq::recv_flags::none);
        if (!recv_result) {
            std::cerr << "Error receiving reply for captain data." << std::endl;
            return false;
        }

        std::string reply_str(static_cast<char*>(reply.data()), reply.size());
        nlohmann::json response = nlohmann::json::parse(reply_str);

        return response["status"] == "success";
    }

    bool receiveControlAction(std::vector<spaceship>& ships) {
        if (!connected) {
            std::cerr << "Not connected to firmware server." << std::endl;
            return false;
        }

        nlohmann::json request;
        request["type"] = "get_control_actions";

        std::string request_str = request.dump();
        zmq::message_t message(request_str.size());
        memcpy(message.data(), request_str.data(), request_str.size());

        auto send_result = socket.send(message, zmq::send_flags::none);
        if (!send_result) {
            std::cerr << "Error sending control request to firmware server." << std::endl;
            return false;
        }

        zmq::message_t reply;
        auto recv_result = socket.recv(reply, zmq::recv_flags::none);
        if (!recv_result) {
            std::cerr << "Error receiving control actions from firmware server." << std::endl;
            return false;
        }

        try {
            std::string reply_str(static_cast<char*>(reply.data()), reply.size());
            nlohmann::json response = nlohmann::json::parse(reply_str);

            if (response["status"] != "success") {
                std::cerr << "Server returned error: " << response["message"].get<std::string>() << std::endl;
                return false;
            }

            auto& actions = response["actions"];
            if (!actions.is_array()) {
                std::cerr << "Invalid control actions format." << std::endl;
                return false;
            }

            for (const auto& action : actions) {
                int id = action["ship_id"].get<int>();

                // Find the ship with the matching ID
                auto ship_it = std::find_if(ships.begin(), ships.end(),
                                           [id](const spaceship& s) { return s.id == id; });

                if (ship_it != ships.end()) {
                    // Update thrust
                    if (action.contains("thrust")) {
                        auto& thrust = action["thrust"];
                        ship_it->thrustX = thrust[0].get<float>();
                        ship_it->thrustY = thrust[1].get<float>();
                        ship_it->thrustZ = thrust[2].get<float>();
                    }

                    // Update other control parameters as needed
                    // For example, orientation, engine state, etc.
                }
            }

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error processing control actions: " << e.what() << std::endl;
            return false;
        }
    }
};

inline bool firmwareClient::sendSimuationState(const std::vector<Body> &bodies, const std::vector<spaceship> &ships) {
    if (!connected) {
        std::cerr << "Not connected to firmware server." << std::endl;
        return false;
    }
    nlohmann::json state;

    state["spacecraft"] = nlohmann::json::array();
    for (const auto& ship : ships) {
        nlohmann::json ship_data;
        ship_data["id"] = ship.id;  // Pass value, not address
        ship_data["position"] = { ship.x, ship.y, ship.z };  // Use position vector
        ship_data["velocity"] = { ship.vx, ship.vy, ship.vz };  // Use velocity vector
        ship_data["mass"] = ship.mass;
        ship_data["thrust"] = { ship.thrustX, ship.thrustY, ship.thrustZ };
        ship_data["fuel"] = ship.fuel;

        state["spacecraft"].push_back(ship_data);
    }
    // Add celestial body data
    state["celestial_bodies"] = nlohmann::json::array();
    for (const auto& body : bodies) {
        nlohmann::json body_data;
        body_data["id"] = &body - &bodies[0]; // Index
        body_data["position"] = { body.x, body.y, body.z };  // Use position vector
        body_data["velocity"] = { body.vx, body.vy, body.vz };  // Use velocity vector
        body_data["mass"] = body.mass;
        body_data["radius"] = body.radius;
        // Add other relevant body properties

        state["celestial_bodies"].push_back(body_data);
    }

    nlohmann::json request;
    request["type"] = "update_state";  // Fixed typo (removed leading space)
    request["data"] = state;

    std::string request_str = request.dump();
    zmq::message_t message(request_str.size());
    memcpy(message.data(), request_str.data(), request_str.size());

    auto send_result = socket.send(message, zmq::send_flags::none);
    if (!send_result) {
        std::cerr << "Error sending message to firmware server." << std::endl;
        return false;
    }
    zmq::message_t reply;
    auto recv_result = socket.recv(reply, zmq::recv_flags::none);
    if (!recv_result) {
        std::cerr << "Error receiving reply from firmware server." << std::endl;
        return false;
    }

    std::string reply_str(static_cast<char*>(reply.data()), reply.size());
    nlohmann::json response = nlohmann::json::parse(reply_str);

    return response["status"] == "success";
}
#endif //FIRMWARE_SERVER_CUH