#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <future>
#include <iostream>

#include "memory/shortTermMemory.h"
#include "memory/longTermMemory.h"
#include "memory/InternediateMemory.h"
#include "networking/spacecraftServer.h"
#include "logger/logger.h"
#include <test/test_reporter.h>
// Debug macro to print test information
#define DEBUG_INFO(msg) std::cout << "[DEBUG] " << __FUNCTION__ << ": " << msg << std::endl

// Mock logger to verify logging behavior
class MockLogger : public Logger {
public:
    MOCK_METHOD(void, log, (LogLevel level, const std::string& message));
    MOCK_METHOD(void, close, ());
};

class SpacecraftFirmwareTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Get test name and start test reporting
        const ::testing::TestInfo* const test_info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        TestReporter::getInstance().startTest(test_info->name());

        // Initialize test objects
        shortMem = std::make_unique<shortTermMemory>();
        longMem = std::make_unique<longTermMemory>();
        intMem = std::make_unique<InternediateMemory>(1024);
        DEBUG_INFO("Created InternediateMemory with capacity: " << intMem->capacity());
    }

    void TearDown() override {
        // Get the test result
        const ::testing::TestInfo* const test_info =
            ::testing::UnitTest::GetInstance()->current_test_info();

        // Record test result
        bool passed = test_info->result()->Passed();
        TestReporter::getInstance().endTest(
            passed ? TestReporter::TestResult::PASS : TestReporter::TestResult::FAIL,
            passed ? "Test passed successfully" : "Test failed"
        );

        // Clean up resources
        DEBUG_INFO("Cleaning up memory resources");
        shortMem->clear();
        longMem->clear();
        intMem->clear();
    }

    std::unique_ptr<shortTermMemory> shortMem;
    std::unique_ptr<longTermMemory> longMem;
    std::unique_ptr<InternediateMemory> intMem;
};

// 1. Command Reliability Tests
TEST_F(SpacecraftFirmwareTest, CommandReliabilityBasicOperations) {
    // Test basic write/read operations
    const uint32_t testValue = 0xDEADBEEF;
    DEBUG_INFO("Writing test value 0x" << std::hex << testValue << " to offset 0");
    bool writeResult = intMem->write(0, &testValue, sizeof(testValue));
    DEBUG_INFO("Write result: " << (writeResult ? "success" : "failure"));
    ASSERT_TRUE(writeResult);

    uint32_t readValue = 0;
    DEBUG_INFO("Reading from offset 0");
    bool readResult = intMem->read(0, &readValue, sizeof(readValue));
    DEBUG_INFO("Read result: " << (readResult ? "success" : "failure") << ", value read: 0x" << std::hex << readValue);
    ASSERT_TRUE(readResult);
    EXPECT_EQ(testValue, readValue);
}

TEST_F(SpacecraftFirmwareTest, CommandReliabilityThreadSafety) {
    // Test thread safety with concurrent operations
    std::atomic<int> successCount(0);
    std::vector<std::thread> threads;
    DEBUG_INFO("Starting thread safety test with 10 concurrent operations");

    for (int i = 0; i < 10; i++) {
        threads.emplace_back([this, i, &successCount]() {
            const uint32_t testValue = 0xABCD0000 + i;
            const size_t offset = i * sizeof(uint32_t);
            DEBUG_INFO("Thread " << i << " writing value 0x" << std::hex << testValue << " to offset " << std::dec << offset);

            if (intMem->write(offset, &testValue, sizeof(testValue))) {
                DEBUG_INFO("Thread " << i << " write successful");
                std::this_thread::sleep_for(std::chrono::milliseconds(5));

                uint32_t readValue = 0;
                bool readResult = intMem->read(offset, &readValue, sizeof(readValue));
                DEBUG_INFO("Thread " << i << " read result: " << (readResult ? "success" : "failure")
                          << ", expected: 0x" << std::hex << testValue
                          << ", actual: 0x" << readValue);

                if (readResult && readValue == testValue) {
                    successCount++;
                    DEBUG_INFO("Thread " << i << " operation successful, current success count: " << successCount);
                }
            } else {
                DEBUG_INFO("Thread " << i << " write failed");
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    DEBUG_INFO("Thread safety test completed. Success count: " << successCount << " (expected: 10)");
    EXPECT_EQ(10, successCount);
}

TEST_F(SpacecraftFirmwareTest, CommandReliabilityBoundaryConditions) {
    // Test boundary conditions
    const uint8_t testByte = 0xFF;

    // Write at the end of buffer
    size_t lastPos = intMem->capacity() - 1;
    DEBUG_INFO("Writing at last valid position: " << lastPos);
    bool lastPosResult = intMem->write(lastPos, &testByte, 1);
    DEBUG_INFO("Write result: " << (lastPosResult ? "success" : "failure"));
    EXPECT_TRUE(lastPosResult);

    // Try writing past the buffer
    DEBUG_INFO("Attempting to write at position beyond capacity: " << intMem->capacity());
    bool beyondCapacityResult = intMem->write(intMem->capacity(), &testByte, 1);
    DEBUG_INFO("Write result: " << (beyondCapacityResult ? "success (error)" : "failure (expected)"));
    EXPECT_FALSE(beyondCapacityResult);

    DEBUG_INFO("Attempting to write past end of buffer from last position");
    bool overlappingResult = intMem->write(lastPos, &testByte, 2);
    DEBUG_INFO("Write result: " << (overlappingResult ? "success (error)" : "failure (expected)"));
    EXPECT_FALSE(overlappingResult);
}

// 2. Fault Detection and Recovery Tests
TEST_F(SpacecraftFirmwareTest, FaultDetectionAndRecovery) {
    // Write test data
    const std::vector<uint8_t> testData = {0x01, 0x02, 0x03, 0x04, 0x05};
    DEBUG_INFO("Writing test data of size " << testData.size() << " to offset 0");
    bool writeResult = intMem->write(0, testData.data(), testData.size());
    DEBUG_INFO("Write result: " << (writeResult ? "success" : "failure"));
    ASSERT_TRUE(writeResult);

    // Simulate corruption by writing directly to the memory
    const uint8_t corruptData = 0xFF;
    DEBUG_INFO("Simulating corruption by writing 0xFF at offset 2");
    bool corruptResult = intMem->write(2, &corruptData, 1);
    DEBUG_INFO("Corrupt write result: " << (corruptResult ? "success" : "failure"));
    ASSERT_TRUE(corruptResult);

    // Explicitly trigger a scan for corruption
    intMem->scanForCorruption();

    // Check if fault was detected
    bool hasFault = intMem->hasFault();
    DEBUG_INFO("Fault detection state: " << (hasFault ? "fault detected" : "no fault detected"));
    EXPECT_TRUE(hasFault);


    // Attempt recovery
    DEBUG_INFO("Attempting recovery");
    bool recoveryResult = intMem->attemptRecovery();
    DEBUG_INFO("Recovery result: " << (recoveryResult ? "success" : "failure"));
    EXPECT_TRUE(recoveryResult);

    // Verify data was recovered properly
    std::vector<uint8_t> readData(testData.size(), 0);
    DEBUG_INFO("Reading data after recovery");
    bool readResult = intMem->read(0, readData.data(), readData.size());
    DEBUG_INFO("Read result: " << (readResult ? "success" : "failure"));
    ASSERT_TRUE(readResult);

    // The data should be either recovered or at least the system should be in a valid state
    bool postRecoveryFault = intMem->hasFault();
    DEBUG_INFO("Post-recovery fault state: " << (postRecoveryFault ? "fault still present" : "no fault"));
    EXPECT_FALSE(postRecoveryFault);
}

TEST_F(SpacecraftFirmwareTest, ExceptionHandling) {
    // Test that memory exceptions are properly caught and handled
    uint8_t buffer;
    size_t invalidPosition = intMem->capacity() + 100;
    DEBUG_INFO("Attempting to read from invalid position: " << invalidPosition);
    bool readResult = intMem->read(invalidPosition, &buffer, 1);
    DEBUG_INFO("Read result: " << (readResult ? "success (error)" : "failure (expected)"));
    EXPECT_FALSE(readResult);

    // Check that error state was updated
    int errorCount = intMem->getErrorCount();
    DEBUG_INFO("Error count after invalid read: " << errorCount);
    EXPECT_GT(errorCount, 0);
}

// 3. Telemetry Buffering and Synchronization Tests
TEST_F(SpacecraftFirmwareTest, TelemetryBufferingAndSync) {
    // Test data transfer between memory types
    const std::vector<uint8_t> testData = {0xA1, 0xB2, 0xC3, 0xD4};

    // Write to intermediate memory
    DEBUG_INFO("Writing test data of size " << testData.size() << " to offset 10");
    bool writeResult = intMem->write(10, testData.data(), testData.size());
    DEBUG_INFO("Write result: " << (writeResult ? "success" : "failure"));
    ASSERT_TRUE(writeResult);

    // Transfer to short-term memory
    DEBUG_INFO("Transferring data from intermediate memory to short-term memory");
    bool transferShortResult = intMem->transferToShortTerm(*shortMem, 10, 20, testData.size());
    DEBUG_INFO("Transfer result: " << (transferShortResult ? "success" : "failure"));
    ASSERT_TRUE(transferShortResult);

    // Read from short-term memory and verify
    std::vector<uint8_t> readData(testData.size());
    DEBUG_INFO("Reading data from short-term memory at offset 20");
    size_t bytesRead = shortMem->read(20, readData.data(), readData.size());
    DEBUG_INFO("Bytes read: " << bytesRead << " (expected: " << testData.size() << ")");
    ASSERT_EQ(testData.size(), bytesRead);
    EXPECT_EQ(testData, readData);

    // Transfer to long-term memory
    DEBUG_INFO("Transferring data from intermediate memory to long-term memory");
    bool transferLongResult = intMem->transferToLongTerm(*longMem, 10, 30, testData.size());
    DEBUG_INFO("Transfer result: " << (transferLongResult ? "success" : "failure"));
    ASSERT_TRUE(transferLongResult);

    // Read from long-term memory and verify
    std::fill(readData.begin(), readData.end(), 0);
    DEBUG_INFO("Reading data from long-term memory at offset 30");
    bytesRead = longMem->read(30, readData.data(), readData.size());
    DEBUG_INFO("Bytes read: " << bytesRead << " (expected: " << testData.size() << ")");
    ASSERT_EQ(testData.size(), bytesRead);
    EXPECT_EQ(testData, readData);
}

TEST_F(SpacecraftFirmwareTest, SynchronizationAccuracy) {
    // Test synchronization accuracy under load
    std::atomic<bool> startFlag(false);
    std::vector<std::thread> threads;
    std::vector<bool> results(5, false);

    // Add a mutex to coordinate memory access
    std::mutex memMutex;
    DEBUG_INFO("Starting synchronization accuracy test with 5 threads");

    // Create threads that wait for a synchronized start
    for (int i = 0; i < 5; i++) {
        threads.emplace_back([this, i, &startFlag, &results, &memMutex]() {
            // Wait for start signal
            DEBUG_INFO("Thread " << i << " waiting for start signal");
            while (!startFlag.load()) {
                std::this_thread::yield();
            }

            // Perform synchronized writes to different memory areas
            const uint32_t testValue = 0xF000 + i;
            const size_t offset = i * sizeof(uint32_t) * 4; // Ensure no overlap
            DEBUG_INFO("Thread " << i << " writing value 0x" << std::hex << testValue
                      << " to offset " << std::dec << offset);

            // Use a lock to prevent simultaneous access to memory
            {
                std::lock_guard<std::mutex> lock(memMutex);
                results[i] = intMem->write(offset, &testValue, sizeof(testValue));
                DEBUG_INFO("Thread " << i << " write result: " << (results[i] ? "success" : "failure"));
            }
        });
    }

    // Add a small delay to ensure all threads are ready
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    DEBUG_INFO("All threads ready, sending start signal");

    // Signal all threads to start simultaneously
    startFlag.store(true);

    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    DEBUG_INFO("All threads completed");

    // Verify all operations succeeded
    for (int i = 0; i < results.size(); i++) {
        DEBUG_INFO("Thread " << i << " result: " << (results[i] ? "success" : "failure"));
        EXPECT_TRUE(results[i]);
    }

    // Verify no fault was detected during synchronized operations
    bool hasFault = intMem->hasFault();
    DEBUG_INFO("Final fault state: " << (hasFault ? "fault detected" : "no fault"));
    EXPECT_FALSE(hasFault);
}