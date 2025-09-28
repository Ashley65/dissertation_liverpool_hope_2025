//
// Created by DevAccount on 28/03/2025.
//

#ifndef SOLAR_SYSTEM_FACTORY_H
#define SOLAR_SYSTEM_FACTORY_H
// include/core/solar_system_factory.cuh
#ifndef SOLAR_SYSTEM_FACTORY_CUH
#define SOLAR_SYSTEM_FACTORY_CUH
#pragma once


#include <string>
#include <vector>

#include "core/ECS.cuh"
#include "core/physics_components.cuh"

class SolarSystemFactory {
public:
    static void createSolarSystem(EntityManager& entityManager) {
        const float AU = 1.496e11f; // Astronomical Unit in meters

        // Create Sun
        Entity sun = entityManager.createEntity();
        entityManager.addComponent<TransformComponent>(sun, 0.0, 0.0, 0.0);
        entityManager.addComponent<VelocityComponent>(sun, 0.0, 0.0, 0.0);
        entityManager.addComponent<AccelerationComponent>(sun, 0.0, 0.0, 0.0);
        entityManager.addComponent<MassComponent>(sun, 1.989e30f, 6.957e8f);
        entityManager.addComponent<CelestialBodyComponent>(sun, "Sun", false);

        // Create planets
        createPlanet(entityManager, "Mercury", 3.3011e23f, 2.4397e6f, 0.387 * AU, 0.0, 0.122f);
        createPlanet(entityManager, "Venus", 4.8675e24f, 6.0518e6f, 0.723 * AU, 0.0, 0.059f);
        createPlanet(entityManager, "Earth", 5.972e24f, 6.371e6f, 1.0 * AU, 0.0);
        createPlanet(entityManager, "Mars", 6.4171e23f, 3.3895e6f, 1.524 * AU, 0.0, 0.032f);
        createPlanet(entityManager, "Jupiter", 1.8982e27f, 6.9911e7f, 5.2 * AU, 0.0, 0.022f);
        createPlanet(entityManager, "Saturn", 5.6834e26f, 5.8232e7f, 9.58 * AU, 0.0, 0.043f);
        createPlanet(entityManager, "Uranus", 8.6810e25f, 2.5362e7f, 19.2 * AU, 0.0, 0.013f);
        createPlanet(entityManager, "Neptune", 1.02413e26f, 2.4622e7f, 30.05 * AU, 0.0, 0.011f);
    }

private:
    static Entity createPlanet(EntityManager& entityManager, const std::string& name,
                              double mass, double radius, double distance, double z = 0,
                              double eccentricity = 0.0) {
        Entity planet = entityManager.createEntity();

        // Set position (on x-axis by default)
        entityManager.addComponent<TransformComponent>(planet, distance, 0.0, z);

        // Calculate orbital velocity (circular orbit approximation)
        const double G = 6.67430e-11;
        const double sunMass = 1.989e30;
        double v = std::sqrt(G * sunMass / distance) * (1.0 + eccentricity);

        // Set velocity perpendicular to position vector (y-direction)
        entityManager.addComponent<VelocityComponent>(planet, 0.0, v, 0.0);
        entityManager.addComponent<AccelerationComponent>(planet, 0.0, 0.0, 0.0);
        entityManager.addComponent<MassComponent>(planet, mass, radius);
        entityManager.addComponent<CelestialBodyComponent>(planet, name, true);

        return planet;
    }
};
#endif
#endif //SOLAR_SYSTEM_FACTORY_H
