
#ifndef TEST_REPORTER_H
#define TEST_REPORTER_H

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <map>
#include <mutex>

class TestReporter {
public:
    enum class TestResult {
        PASS,
        FAIL,
        SKIPPED
    };

    struct TestEntry {
        std::string testName;
        TestResult result;
        std::chrono::milliseconds duration;
        std::string details;
    };

    static TestReporter& getInstance() {
        static TestReporter instance;
        return instance;
    }

    void startTest(const std::string& testName) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_currentTest = testName;
        m_startTime = std::chrono::steady_clock::now();
    }

    void endTest(TestResult result, const std::string& details = "") {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_startTime);

        m_testEntries.push_back({
            m_currentTest,
            result,
            duration,
            details
        });

        switch (result) {
            case TestResult::PASS: m_passCount++; break;
            case TestResult::FAIL: m_failCount++; break;
            case TestResult::SKIPPED: m_skipCount++; break;
        }
    }

    void generateReport(const std::string& filename = "spacecraft_test_report.txt") {
        std::ofstream reportFile(filename);

        if (!reportFile.is_open()) {
            std::cerr << "Failed to open report file: " << filename << std::endl;
            return;
        }

        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);

        reportFile << "=================================================\n";
        reportFile << "SPACECRAFT FIRMWARE TEST REPORT\n";
        reportFile << "=================================================\n";
        reportFile << "Date: " << std::ctime(&now_time_t);
        reportFile << "Total Tests: " << m_testEntries.size() << "\n";
        reportFile << "Passed: " << m_passCount << "\n";
        reportFile << "Failed: " << m_failCount << "\n";
        reportFile << "Skipped: " << m_skipCount << "\n";
        reportFile << "=================================================\n\n";

        reportFile << "TEST DETAILS:\n";
        reportFile << "-------------------------------------------------\n";

        for (const auto& entry : m_testEntries) {
            reportFile << "Test: " << entry.testName << "\n";
            reportFile << "Result: " << resultToString(entry.result) << "\n";
            reportFile << "Duration: " << entry.duration.count() << "ms\n";

            if (!entry.details.empty()) {
                reportFile << "Details: " << entry.details << "\n";
            }

            reportFile << "-------------------------------------------------\n";
        }

        reportFile << "\nEnd of Report\n";
        reportFile.close();

        std::cout << "Test report generated: " << filename << std::endl;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_testEntries.clear();
        m_passCount = 0;
        m_failCount = 0;
        m_skipCount = 0;
    }

private:
    TestReporter() : m_passCount(0), m_failCount(0), m_skipCount(0) {}

    std::string resultToString(TestResult result) {
        switch (result) {
            case TestResult::PASS: return "PASS";
            case TestResult::FAIL: return "FAIL";
            case TestResult::SKIPPED: return "SKIPPED";
            default: return "UNKNOWN";
        }
    }

    std::vector<TestEntry> m_testEntries;
    std::string m_currentTest;
    std::chrono::steady_clock::time_point m_startTime;
    size_t m_passCount;
    size_t m_failCount;
    size_t m_skipCount;
    std::mutex m_mutex;
};
#endif //TEST_REPORTER_H
