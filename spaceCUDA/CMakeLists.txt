cmake_minimum_required(VERSION 3.30)
project(spaceCUDA LANGUAGES CXX CUDA)



set(CMAKE_CUDA_STANDARD 20)

# include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
# Find OpenGL, GLFW, and GLEW
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(Threads REQUIRED)
find_package(cppzmq CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Catch2 CONFIG REQUIRED)
find_package(GTest REQUIRED)






# Include directories
include_directories(${GLEW_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS})

set(SOURCES
        src/main.cu
        include/physics/Body.cuh
        src/physics/physics_engine.cu
        include/physics/physics_engine.cuh
        include/core/ECS.cuh
        include/core/physics_components.cuh
        include/core/physics_system.cuh
        include/core/device_data_manager.cuh
        include/core/dataChecker.cuh
        include/core/simulation_manager.cuh
        include/physics/solar_system_factory.cuh
        src/physics/visualisation/opengl_renderer.cu
        include/physics/visualisation/opengl_renderer.cuh
        include/physics/NetworkingCLient.h
        include/physics/spaceship.cuh
        src/test_physics_engine.cu
        include/ai/AIDataPacket.cuh
)

add_executable(spaceCUDA ${SOURCES})

# Link libraries \w OpenMP and Flecs
target_link_libraries(spaceCUDA PRIVATE
        OpenGL::GL
        glfw
        GLEW::GLEW
        Threads::Threads
)

target_link_libraries(spaceCUDA PRIVATE cppzmq cppzmq-static)
target_link_libraries(spaceCUDA PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(spaceCUDA PRIVATE Catch2::Catch2 Catch2::Catch2WithMain)
target_link_libraries(spaceCUDA PRIVATE GTest::GTest GTest::Main)

set_target_properties(spaceCUDA PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
