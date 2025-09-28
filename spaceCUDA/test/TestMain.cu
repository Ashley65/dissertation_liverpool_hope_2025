#include <gtest/gtest.h>
#include "../include/core/simulation_manager.cuh"

TEST(SimulationManagerTest, InitializeSolarSystem) {
    SimulationManager manager(9000);
    const auto& bodies = manager.getBodies();

    ASSERT_EQ(bodies.size(), 9000); // Ensure 9 bodies are initialized
    EXPECT_FLOAT_EQ(bodies[0].mass, 1.989e30f); // Check Sun's mass
    EXPECT_FLOAT_EQ(bodies[3].vx, 0.0f); // Check Earth's velocity in x
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}