cmake_minimum_required(VERSION 3.30)
project(spacecraftFirmware)

set(CMAKE_CXX_STANDARD 20)
find_package(cppzmq CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Catch2 CONFIG REQUIRED)
find_package(GTest REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(SOURCES
        src/main.cpp
        src/networking/spacecraftServer.cpp
        include/networking/spacecraftServer.h
        include/packagem.h
        src/command/command.cpp
        include/commad/command.h
        src/missionContext/MissionContext.cpp
        include/missionContext/MissionContext.h
        src/missionContext/SpacecraftStatus.cpp
        include/missionContext/SpacecraftStatus.h
        src/logger/logger.cpp
        include/logger/logger.h
        src/memory/InternediateMemory.cpp
        include/memory/InternediateMemory.h
        include/memory/shortTermMemory.h
        include/memory/longTermMemory.h
        include/logger/LogMacros.h
        src/safety/safetyManager.cpp
        include/safety/safetyManager.h
        src/memory/longTermMemory.cpp
        src/memory/shortTermMemory.cpp
        src/Test_spacecraftFirmware.cpp
        include/test/test_reporter.h

)

add_executable(spacecraftFirmware ${SOURCES} )

target_link_libraries(spacecraftFirmware PRIVATE cppzmq cppzmq-static)
target_link_libraries(spacecraftFirmware PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(spacecraftFirmware PRIVATE Catch2::Catch2 Catch2::Catch2WithMain)
target_link_libraries(spacecraftFirmware PRIVATE GTest::GTest GTest::Main)