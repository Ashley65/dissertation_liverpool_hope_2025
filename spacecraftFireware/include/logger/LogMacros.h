//
// Created by DevAccount on 30/04/2025.
//

#ifndef LOGMACROS_H
#define LOGMACROS_H
#pragma once

#include "logger/Logger.h"

#define LOG_DEBUG(message) Logger::getInstance().log(LogLevel::DEBUG, message)
#define LOG_INFO(message) Logger::getInstance().log(LogLevel::INFO, message)
#define LOG_WARNING(message) Logger::getInstance().log(LogLevel::WARNING, message)
#define LOG_ERROR(message) Logger::getInstance().log(LogLevel::ERROR, message)

#endif //LOGMACROS_H
