//
// Created by DevAccount on 04/04/2025.
//

#ifndef LOGGER_H
#define LOGGER_H
#pragma once
#include <packagem.h>
#include "logger.h"
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};


class Logger {
public:
    static Logger& getInstance();

    void init(const std::string& logFilePath);
    void setLogLevel(LogLevel level);
    void log(LogLevel level, const std::string& message);

    // Convenience methods
    void debug(const std::string& message);
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);

    void close();

private:
    Logger();
    ~Logger();

    static std::string getLevelString(LogLevel level);

    static std::string getCurrentTimestamp();

    LogLevel currentLevel;
    std::ofstream logFile;
    std::mutex logMutex;
    bool initialised;
};

// Define log macros that use the Logger singleton
#define LOG_DEBUG(message) Logger::getInstance().log(LogLevel::DEBUG, message)
#define LOG_INFO(message) Logger::getInstance().log(LogLevel::INFO, message)
#define LOG_WARNING(message) Logger::getInstance().log(LogLevel::WARNING, message)
#define LOG_ERROR(message) Logger::getInstance().log(LogLevel::ERROR, message)
// Undefine the old macros if they exist
#ifdef LOG_INFO
#undef LOG_INFO
#endif

#ifdef LOG_ERROR
#undef LOG_ERROR
#endif

#ifdef LOG_WARNING
#undef LOG_WARNING
#endif

#ifdef LOG_DEBUG
#undef LOG_DEBUG

#endif



#endif //LOGGER_H
