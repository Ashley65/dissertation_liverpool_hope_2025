//
// Created by DevAccount on 10/04/2025.
//

#ifndef INTERNEDIATEMEMORY_H
#define INTERNEDIATEMEMORY_H

#pragma once
#include <packagem.h>
#include <logger/logger.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <stdexcept>

// Forward declarations
class shortTermMemory;
class longTermMemory;

// Memory fault exception types
class MemoryAccessException : public std::runtime_error {
public:
    explicit MemoryAccessException(const std::string& what) : std::runtime_error(what) {}
};

class MemoryCorruptionException : public std::runtime_error {
public:
    explicit MemoryCorruptionException(const std::string& what) : std::runtime_error(what) {}
};

class InternediateMemory {
private:
    std::vector<uint8_t> buffer;
    mutable std::mutex accessMutex;  // Mark mutex as mutable so it can be used in const methods
    mutable std::atomic<bool> faultDetected{false};  // Mark atomic as mutable
    mutable std::atomic<uint32_t> errorCount{0};     // Mark atomic as mutable
    Logger& logger;

    // Redundancy for critical data
    struct MemoryCheckpoint {
        uint32_t checksum;
        std::vector<uint8_t> backupData;
        uint64_t timestamp;
    };

    std::vector<MemoryCheckpoint> checkpoints;

    static uint32_t calculateChecksum(const std::vector<uint8_t>& data);
    bool verifyIntegrity();
    void createCheckpoint();
    bool restoreFromCheckpoint();

public:
    InternediateMemory();
    explicit InternediateMemory(size_t initialCapacity);
    ~InternediateMemory();

    // Memory operations with safety measures
    bool write(size_t offset, const void* data, size_t length);
    bool read(size_t offset, void* destination, size_t length) const;

    bool scanForCorruption();

    // Data transfer between memories
    bool transferToShortTerm(shortTermMemory& target, size_t srcOffset, size_t dstOffset, size_t length);
    bool transferToLongTerm(longTermMemory& target, size_t srcOffset, size_t dstOffset, size_t length);

    // Fault handling
    bool hasFault() const { return faultDetected.load(); }
    bool attemptRecovery();
    void resetErrorState();
    uint32_t getErrorCount() const { return errorCount.load(); }

    // Memory management
    bool resize(size_t newSize);
    size_t capacity() const { return buffer.size(); }
    void clear();
};

#endif //INTERNEDIATEMEMORY_H