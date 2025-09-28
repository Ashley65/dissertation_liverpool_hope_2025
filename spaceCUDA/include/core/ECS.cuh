//
// Created by DevAccount on 27/03/2025.
//

#ifndef ECS_CUH
#define ECS_CUH
#pragma once
#include <vector>
#include <unordered_map>
#include <typeindex>
#include <memory>
#include <functional>

// Entity ID type (size_t)
using Entity = std::size_t;

// Component interface
struct Component {
    virtual ~Component() = default;
};


class EntityManager {
  private:
    // Map from type_index to vector of components
    std::unordered_map<std::type_index, std::vector<std::unique_ptr<Component>>> components;

    // Map entity to its components indices
    std::unordered_map<Entity, std::unordered_map<std::type_index, size_t>> entityComponentMap;

    // Next entity ID



  public:
    Entity nextEntity = 0;
     // Create a new entity


     Entity createEntity() {
            return nextEntity++;
     }

     // Add a component to an entity
     template <typename T, typename... Args>
     T* addComponent(Entity entity, Args&&... args) {
         static_assert(std::is_base_of<Component, T>::value, "Type must derive from Component");

         auto& componentArray = components[std::type_index(typeid(T))];
         auto component = std::make_unique<T>(std::forward<Args>(args)...);
         T* componentPtr = component.get();

         entityComponentMap[entity][std::type_index(typeid(T))] = componentArray.size();
         componentArray.push_back(std::move(component));

         return componentPtr;
     }

    // Get a component from an entity
    template <typename T>
    T* getComponent(Entity entity) {
         static_assert(std::is_base_of<Component, T>::value, "Type must derive from Component");

         auto typeIndex = std::type_index(typeid(T));
         if (entityComponentMap.find(entity) == entityComponentMap.end() ||
             entityComponentMap[entity].find(typeIndex) == entityComponentMap[entity].end())
             return nullptr;

         auto index = entityComponentMap[entity][typeIndex];
         return static_cast<T*>(components[typeIndex][index].get());

     }
};








#endif //ECS_CUH
