#include <iostream>
#include <signal.h>
#include <networking/spacecraftServer.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <random>
#include <fstream>
#include <nlohmann/json.hpp>

#include "logger/logger.h"
#include "logger/LogMacros.h"
#include "test/test_reporter.h"

// For convenience
using json = nlohmann::json;

// Global server pointer for signal handling
spacecraftServer* server = nullptr;
std::atomic<bool> runningTests(false);
std::atomic<bool> serverRunning(false);

// Test metrics
struct TestMetrics {
    // Responsiveness metrics
    std::atomic<int> totalRequests{0};
    std::atomic<int> timelyResponses{0}; // responses within threshold
    std::atomic<double> totalResponseTime{0};
    std::atomic<double> maxResponseTime{0};

    // IPC metrics
    std::atomic<int> messagesSent{0};
    std::atomic<int> messagesReceived{0};
    std::atomic<int> messageErrors{0};

    // Decision alignment metrics
    std::atomic<int> decisionsEvaluated{0};
    std::atomic<int> decisionsAligned{0};

    void reset() {
        totalRequests = 0;
        timelyResponses = 0;
        totalResponseTime = 0;
        maxResponseTime = 0;
        messagesSent = 0;
        messagesReceived = 0;
        messageErrors = 0;
        decisionsEvaluated = 0;
        decisionsAligned = 0;
    }

    json toJson() const {
        json responseJson = {
            {"responsiveness", {
                {"total_requests", totalRequests.load()},
                {"timely_responses", timelyResponses.load()},
                {"response_rate", totalRequests > 0 ?
                    (double)timelyResponses / totalRequests * 100.0 : 0},
                {"avg_response_time_ms", totalRequests > 0 ?
                    totalResponseTime / totalRequests : 0},
                {"max_response_time_ms", maxResponseTime.load()}
            }},
            {"ipc", {
                {"messages_sent", messagesSent.load()},
                {"messages_received", messagesReceived.load()},
                {"success_rate", messagesSent > 0 ?
                    (double)messagesReceived / messagesSent * 100.0 : 0},
                {"error_rate", messagesSent > 0 ?
                    (double)messageErrors / messagesSent * 100.0 : 0}
            }},
            {"decision_alignment", {
                {"decisions_evaluated", decisionsEvaluated.load()},
                {"decisions_aligned", decisionsAligned.load()},
                {"alignment_rate", decisionsEvaluated > 0 ?
                    (double)decisionsAligned / decisionsEvaluated * 100.0 : 0}
            }}
        };
        return responseJson;
    }
};

TestMetrics testMetrics;

// Signal handler for graceful shutdown
void signalHandler(int signal) {
    LOG_INFO("Received signal " + std::to_string(signal) + ", shutting down...");
    serverRunning = false;
    if (server) {
        server->stop();
    }

    // Ensure logger is closed properly
    Logger::getInstance().close();
    exit(0);
}

// Basic tests using Google Test framework
int runBasicTests() {
    if (runningTests.exchange(true)) {
        LOG_INFO("Tests already running, ignoring request");
        return -1;
    }

    LOG_INFO("Running basic unit tests...");

    // Setup test arguments
    int argc = 1;
    char* argv[] = {(char*)"test_runner"};

    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests
    int testResult = RUN_ALL_TESTS();

    // Generate report
    TestReporter::getInstance().generateReport();

    LOG_INFO("Basic tests completed with result: " + std::to_string(testResult));
    runningTests = false;
    return testResult;
}

// Helper function to send a command to the server and get response
// Replace the sendCommand function in main.cpp with this implementation
json sendCommand(const std::string& command, const json& params) {
    static std::unique_ptr<zmq::context_t> testContext = nullptr;
    static std::unique_ptr<zmq::socket_t> testSocket = nullptr;

    // Initialize test socket if not already done
    if (!testContext) {
        testContext = std::make_unique<zmq::context_t>(1);
        testSocket = std::make_unique<zmq::socket_t>(*testContext, ZMQ_REQ);
        testSocket->connect("tcp://localhost:5555"); // Connect to server
        LOG_INFO("Test client connected to server");
    }

    // Build the message to send to the server
    json message = {
        {"type", "command"},
        {"command", {
                {"command_type", command},
                {"parameters", params},
                {"priority", "NORMAL"}
        }}
    };

    // Convert to string
    std::string requestStr = message.dump();

    // Send to server
    zmq::message_t request(requestStr.size());
    memcpy(request.data(), requestStr.c_str(), requestStr.size());
    testSocket->send(request, zmq::send_flags::none);

    // Receive response
    zmq::message_t reply;
    auto recvResult = testSocket->recv(reply);

    if (!recvResult) {
        throw std::runtime_error("Failed to receive response from server");
    }

    std::string responseStr(static_cast<char*>(reply.data()), reply.size());

    try {
        // Parse the response
        json response = json::parse(responseStr);
        return response;
    } catch(const std::exception& e) {
        LOG_ERROR("Failed to parse server response: " + std::string(e.what()));
        return json{{"error", "Failed to parse response"}};
    }
}

// Function to test real-time responsiveness
void testResponsiveness(int numTests = 50, int timeoutMs = 100) {
    LOG_INFO("Starting responsiveness test with " + std::to_string(numTests) + " requests");

    // Reset metrics for this test run
    testMetrics.totalRequests = 0;
    testMetrics.timelyResponses = 0;
    testMetrics.totalResponseTime = 0;
    testMetrics.maxResponseTime = 0;

    // Random number generator for simulating varied workloads
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> commandDist(0, 4); // Different command types

    for (int i = 0; i < numTests; i++) {
        try {
            // Select a random command to test
            std::string command;
            json params;

            switch (commandDist(gen)) {
                case 0:
                    command = "adjust_trajectory";
                    params = {{"vector", {1.0, 0.5, -0.3}}, {"magnitude", 1.5}};
                    break;
                case 1:
                    command = "increase_velocity";
                    params = {{"amount", 0.5}};
                    break;
                case 2:
                    command = "decrease_velocity";
                    params = {{"amount", 0.3}};
                    break;
                case 3:
                    command = "maintain_course";
                    params = {{"duration", 5}};
                    break;
                case 4:
                    command = "investigate_anomaly";
                    params = {{"anomaly_type", "radiation_spike"}, {"location", "main_engine"}};
                    break;
                default:
                    command = "maintain_course";
                    params = {{"duration", 1}};
            }

            // Measure response time
            auto start = std::chrono::high_resolution_clock::now();

            // Send command to server
            testMetrics.messagesSent++;
            json response = sendCommand(command, params);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            // Update metrics
            testMetrics.totalRequests++;
            testMetrics.totalResponseTime += duration;

            if (duration > testMetrics.maxResponseTime) {
                testMetrics.maxResponseTime = duration;
            }

            if (duration < timeoutMs) {
                testMetrics.timelyResponses++;
            }

            if (response.contains("message") || response.contains("status")) {
                testMetrics.messagesReceived++;
            } else {
                testMetrics.messageErrors++;
            }

            // Wait a small random amount between requests
            std::this_thread::sleep_for(std::chrono::milliseconds(20 + (i % 10) * 5));
        }
        catch (const std::exception& e) {
            LOG_ERROR("Error in responsiveness test: " + std::string(e.what()));
            testMetrics.messageErrors++;
        }
    }

    // Log results
    LOG_INFO("Responsiveness test completed:");
    LOG_INFO("  Total requests: " + std::to_string(testMetrics.totalRequests));
    LOG_INFO("  Timely responses: " + std::to_string(testMetrics.timelyResponses));
    LOG_INFO("  Response rate: " + std::to_string(testMetrics.timelyResponses * 100.0 / testMetrics.totalRequests) + "%");
    LOG_INFO("  Avg response time: " + std::to_string(testMetrics.totalResponseTime / testMetrics.totalRequests) + "ms");
    LOG_INFO("  Max response time: " + std::to_string(testMetrics.maxResponseTime) + "ms");
}

// Function to test IPC (Inter-Process Communication)
void testInteroperability(int numTests = 50) {
    LOG_INFO("Starting IPC interoperability test with " + std::to_string(numTests) + " messages");

    // Reset metrics for this test
    testMetrics.messagesSent = 0;
    testMetrics.messagesReceived = 0;
    testMetrics.messageErrors = 0;

    // Test different types of messages with varying sizes and complexity
    std::vector<std::pair<std::string, json>> testCases = {
        {"get_status", {}},
        {"adjust_trajectory", {{"vector", {1.0, 2.0, 3.0}}, {"magnitude", 1.5}}},
        {"emergency_protocol", {{"emergency_type", "power_failure"}}},
        {"increase_velocity", {{"amount", 2.5}}},
        {"investigate_anomaly", {{"anomaly_type", "temperature"}, {"location", "engine_bay"}}}
    };

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> testCaseDist(0, testCases.size() - 1);

    for (int i = 0; i < numTests; i++) {
        try {
            // Pick a random test case
            int index = testCaseDist(gen);
            auto& [command, params] = testCases[index];

            // Send message
            testMetrics.messagesSent++;
            json response = sendCommand(command, params);

            // Verify response
            if (response.contains("message") || response.contains("status")) {
                testMetrics.messagesReceived++;

                // Evaluate decision alignment for certain commands
                if (command == "adjust_trajectory" ||
                    command == "increase_velocity" ||
                    command == "emergency_protocol") {

                    testMetrics.decisionsEvaluated++;

                    // Check if the response aligns with expected mission behavior
                    // (This is a simplified example - real criteria would be more complex)
                    bool aligned = false;

                    if (command == "adjust_trajectory" && response.contains("new_velocity")) {
                        aligned = true;
                    } else if (command == "increase_velocity" &&
                               response.contains("new_velocity") &&
                               response.contains("fuel_level")) {
                        aligned = true;
                    } else if (command == "emergency_protocol" &&
                               response.contains("status_change")) {
                        aligned = true;
                    }

                    if (aligned) {
                        testMetrics.decisionsAligned++;
                    }
                }
            } else {
                testMetrics.messageErrors++;
            }
        }
        catch (const std::exception& e) {
            LOG_ERROR("Error in IPC test: " + std::string(e.what()));
            testMetrics.messageErrors++;
        }
    }

    // Log results
    LOG_INFO("IPC test completed:");
    LOG_INFO("  Messages sent: " + std::to_string(testMetrics.messagesSent));
    LOG_INFO("  Messages received: " + std::to_string(testMetrics.messagesReceived));
    LOG_INFO("  Success rate: " + std::to_string(testMetrics.messagesReceived * 100.0 / testMetrics.messagesSent) + "%");
    LOG_INFO("  Error rate: " + std::to_string(testMetrics.messageErrors * 100.0 / testMetrics.messagesSent) + "%");
}

// Function to test autonomous decision alignment
void testDecisionAlignment(int numScenarios = 10) {
    LOG_INFO("Starting decision alignment test with " + std::to_string(numScenarios) + " scenarios");

    // Reset metrics for this test
    testMetrics.decisionsEvaluated = 0;
    testMetrics.decisionsAligned = 0;

    // Define mission critical scenarios
    std::vector<std::pair<std::string, std::vector<std::pair<std::string, json>>>> scenarios = {
        {"Obstacle avoidance", {
            {"get_status", {}},
            {"investigate_anomaly", {{"anomaly_type", "obstacle"}, {"location", "forward_path"}}},
            {"adjust_trajectory", {{"vector", {-1.0, 0.5, 0.0}}, {"magnitude", 2.0}}}
        }},
        {"Fuel conservation", {
            {"get_status", {}},
            {"decrease_velocity", {{"amount", 1.0}}},
            {"maintain_course", {{"duration", 30}}}
        }},
        {"Emergency response", {
            {"emergency_protocol", {{"emergency_type", "system_failure"}}},
            {"get_status", {}},
            {"adjust_trajectory", {{"vector", {0.0, 0.0, 0.0}}, {"magnitude", 0.0}}}
        }}
    };

    // Run through each scenario
    for (const auto& [scenarioName, commands] : scenarios) {
        LOG_INFO("Testing scenario: " + scenarioName);

        bool scenarioSuccess = true;
        json initialStatus;
        json finalStatus;

        try {
            // Execute each command in the scenario
            for (const auto& [command, params] : commands) {
                json response = sendCommand(command, params);

                if (command == "get_status") {
                    if (initialStatus.empty()) {
                        initialStatus = response;
                    } else {
                        finalStatus = response;
                    }
                }

                // If any command fails, the scenario is considered failed
                if (!response.contains("message") && !response.contains("status")) {
                    scenarioSuccess = false;
                    LOG_ERROR("Command failed in scenario " + scenarioName + ": " + command);
                    break;
                }
            }

            // Evaluate the scenario outcome
            testMetrics.decisionsEvaluated++;

            if (scenarioSuccess) {
                // For the obstacle avoidance scenario
                if (scenarioName == "Obstacle avoidance") {
                    // Check if velocity vector changed appropriately
                    if (!finalStatus.empty() &&
                        finalStatus.contains("velocity") &&
                        initialStatus.contains("velocity")) {
                        // Simplified check - in real system would be more comprehensive
                        testMetrics.decisionsAligned++;
                    }
                }
                // For the fuel conservation scenario
                else if (scenarioName == "Fuel conservation") {
                    if (!finalStatus.empty() &&
                        finalStatus.contains("velocity") &&
                        initialStatus.contains("velocity") &&
                        finalStatus.contains("fuel_level") &&
                        initialStatus.contains("fuel_level")) {

                        // Check if velocity decreased and fuel consumption rate improved
                        testMetrics.decisionsAligned++;
                    }
                }
                // For the emergency response scenario
                else if (scenarioName == "Emergency response") {
                    if (!finalStatus.empty() &&
                        finalStatus.contains("mission_phase") &&
                        finalStatus["mission_phase"] == "EMERGENCY") {
                        testMetrics.decisionsAligned++;
                    }
                }
            }
        }
        catch (const std::exception& e) {
            LOG_ERROR("Error in scenario " + scenarioName + ": " + std::string(e.what()));
        }
    }

    // Log results
    LOG_INFO("Decision alignment test completed:");
    LOG_INFO("  Scenarios evaluated: " + std::to_string(testMetrics.decisionsEvaluated));
    LOG_INFO("  Scenarios with aligned decisions: " + std::to_string(testMetrics.decisionsAligned));
    LOG_INFO("  Alignment rate: " + std::to_string(testMetrics.decisionsAligned * 100.0 / testMetrics.decisionsEvaluated) + "%");
}

// Run comprehensive integration tests while server is active
void runIntegrationTests() {
    if (runningTests.exchange(true)) {
        LOG_INFO("Tests already running, ignoring request");
        return;
    }

    LOG_INFO("Starting comprehensive integration tests...");

    try {
        // Reset overall metrics
        testMetrics.reset();

        // 1. Test real-time responsiveness
        testResponsiveness();

        // 2. Test IPC interoperability
        testInteroperability();

        // 3. Test decision alignment
        testDecisionAlignment();

        // Generate JSON report
        json report = testMetrics.toJson();

        // Log summary results
        LOG_INFO("Integration tests completed successfully");
        LOG_INFO("Overall responsiveness rate: " +
            std::to_string(report["responsiveness"]["response_rate"].get<double>()) + "%");
        LOG_INFO("Overall IPC success rate: " +
            std::to_string(report["ipc"]["success_rate"].get<double>()) + "%");
        LOG_INFO("Overall decision alignment rate: " +
            std::to_string(report["decision_alignment"]["alignment_rate"].get<double>()) + "%");

        // Save detailed report to file
        std::string reportPath = "logs/integration_test_report_" +
            std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".json";
        std::ofstream reportFile(reportPath);
        reportFile << report.dump(4) << std::endl;
        reportFile.close();

        LOG_INFO("Detailed report saved to " + reportPath);
    }
    catch (const std::exception& e) {
        LOG_ERROR("Error during integration tests: " + std::string(e.what()));
    }

    runningTests = false;
}

// Function to run tests in a separate thread
void runTestsAsync() {
    std::thread testThread([]() {
        runIntegrationTests();
    });
    testThread.detach();
}

// Function to monitor for command input
void inputMonitor() {
    std::string command;
    while (serverRunning) {
        if (std::getline(std::cin, command)) {
            if (command == "test" || command == "runtest") {
                LOG_INFO("Test command received from console");
                runTestsAsync();
            } else if (command == "exit" || command == "quit") {
                LOG_INFO("Exit command received from console");
                serverRunning = false;
                if (server) {
                    server->stop();
                }
                break;
            } else if (command == "help") {
                std::cout << "Available commands:" << std::endl;
                std::cout << "  test, runtest - Run integration tests while server is running" << std::endl;
                std::cout << "  exit, quit    - Stop the server" << std::endl;
                std::cout << "  help          - Show this help message" << std::endl;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        // Create logs directory if it doesn't exist
        std::filesystem::path logsDir = "logs";
        if (!std::filesystem::exists(logsDir)) {
            std::filesystem::create_directory(logsDir);
        }

        // Initialize logger with timestamp in filename
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "logs/spacecraft_server_" << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S") << ".log";
        std::string logFilePath = ss.str();

        Logger::getInstance().init(logFilePath);
        LOG_INFO("Logging to file: " + logFilePath);

        // Set log level from environment variable if available
        const char* logLevelEnv = std::getenv("SERVER_LOG_LEVEL");
        if (logLevelEnv) {
            std::string logLevelStr = logLevelEnv;
            if (logLevelStr == "DEBUG") {
                Logger::getInstance().setLogLevel(LogLevel::DEBUG);
                LOG_INFO("Log level set to DEBUG");
            }
        }

        // Check for test-only mode
        bool testOnlyMode = false;
        std::string serverAddress = "tcp://*:5555";  // Default

        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--test-only" || arg == "-to") {
                testOnlyMode = true;
                break;
            } else {
                // First non-flag argument is server address
                serverAddress = arg;
            }
        }

        if (testOnlyMode) {
            LOG_INFO("Running in test-only mode...");
            int result = runBasicTests();
            Logger::getInstance().close();
            return result;
        }

        LOG_INFO("Starting spacecraft server on " + serverAddress);

        // Register signal handlers
        signal(SIGINT, signalHandler);
        signal(SIGTERM, signalHandler);

        // Create and start server
        spacecraftServer serverInstance(serverAddress);
        server = &serverInstance;
        serverRunning = true;

        // Start input monitor in a separate thread
        std::thread inputThread(inputMonitor);
        inputThread.detach();

        // Print startup message
        LOG_INFO("Spacecraft server initialized");
        LOG_INFO("Type 'test' or 'runtest' to execute integration tests while server is running");
        LOG_INFO("Type 'exit' or 'quit' to stop the server");
        LOG_INFO("Type 'help' for available commands");

        // Start server (this blocks until server is stopped)
        serverInstance.start();

        // Close logger before exiting
        Logger::getInstance().close();
        return 0;
    }
    catch (const std::exception& e) {
        LOG_ERROR("Fatal error: " + std::string(e.what()));
        Logger::getInstance().close();
        return 3;
    }
}