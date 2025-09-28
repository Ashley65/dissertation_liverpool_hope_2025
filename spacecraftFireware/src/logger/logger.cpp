//
// Created by DevAccount on 04/04/2025.
//

#include <logger/logger.h>

#include <chrono>
#include <iomanip>

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

Logger::Logger() : currentLevel(LogLevel::INFO), initialised(false) {}

Logger::~Logger() {
    close();
}

void Logger::init(const std::string& logFilePath) {
    std::lock_guard<std::mutex> lock(logMutex);
    if (!initialised) {
        logFile.open(logFilePath, std::ios::app);
        initialised = true;
    }
}

void Logger::setLogLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(logMutex);
    currentLevel = level;
}

void Logger::log(LogLevel level, const std::string& message) {
    if (!initialised || level < currentLevel) return;

    std::lock_guard<std::mutex> lock(logMutex);
    logFile << getCurrentTimestamp() << " [" << getLevelString(level) << "] "
            << message << std::endl;
}

void Logger::debug(const std::string& message) {
    log(LogLevel::DEBUG, message);
}

void Logger::info(const std::string& message) {
    log(LogLevel::INFO, message);
}

void Logger::warning(const std::string& message) {
    log(LogLevel::WARNING, message);
}

void Logger::error(const std::string& message) {
    log(LogLevel::ERROR, message);
}

void Logger::close() {
    std::lock_guard<std::mutex> lock(logMutex);
    if (logFile.is_open()) {
        logFile.close();
    }
    initialised = false;
}

std::string Logger::getLevelString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

std::string Logger::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}