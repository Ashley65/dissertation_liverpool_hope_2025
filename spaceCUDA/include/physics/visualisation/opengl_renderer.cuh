//
// Created by DevAccount on 28/03/2025.
//

#ifndef OPENGL_RENDERER_CUH
#define OPENGL_RENDERER_CUH

#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "../physics/Body.cuh"
#include "../core/ECS.cuh"

class OpenGLRenderer {
private:
    GLFWwindow* window;
    GLuint sphereVAO, sphereVBO, sphereEBO;
    GLuint shaderProgram;

    int numVertices;
    int numIndices;

    glm::mat4 projectionMatrix;
    glm::mat4 viewMatrix;

    // Camera settings
    glm::vec3 cameraPos;
    glm::vec3 cameraFront;
    glm::vec3 cameraUp;
    float cameraSpeed;
    float yaw, pitch;

    // Scale factor for celestial visualization
    float visualizationScale;

    // Create sphere mesh
    void createSphere(float radius, unsigned int rings, unsigned int sectors);

    // Shader compilation
    GLuint compileShader(const char* vertexShaderSource, const char* fragmentShaderSource);

public:
    OpenGLRenderer(int width = 1280, int height = 720, const char* title = "N-Body Simulation");
    ~OpenGLRenderer();

    bool init();
    bool shouldClose();
    void beginFrame();
    void endFrame();

    // Draw celestial bodies
    void renderBodies(const std::vector<Body>& bodies, const std::vector<std::string>& names);

    // Handle camera movement
    void processInput();
    void setViewParameters(float distance, float scale);

    // Callback setups
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
};



#endif //OPENGL_RENDERER_CUH
