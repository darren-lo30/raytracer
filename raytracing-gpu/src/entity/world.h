#pragma once

#include "entity_manager.h"
#include "component_manager.h"
#include "system_manager.h"

class EntityHandle;

class World {
private:
	std::shared_ptr<EntityManager> entityManager;
	std::shared_ptr<ComponentManager> componentManager;
	std::shared_ptr<SystemManager> systemManager;
	std::shared_ptr<EventManager> eventManager;

public:
	World();

	//Entities now handle components
	EntityHandle createEntity();
	void destroyEntity(EntityHandle entity);

	// Systems
	template <typename SystemType, typename ... Args>
	std::shared_ptr<SystemType> registerUpdateSystem(Args&& ... args) {
		auto registeredSystem = systemManager->registerUpdateSystem<SystemType>(std::forward<Args>(args)...);
		registeredSystem->setParentWorld(this);
		registeredSystem->setEventManager(eventManager);
		registeredSystem->init();
		return registeredSystem;
	}

	template <typename SystemType, typename... Args>
	std::shared_ptr<SystemType> registerRenderSystem(Args&& ... args) {
		auto registeredSystem = systemManager->registerRenderSystem<SystemType>(std::forward<Args>(args)...);
		registeredSystem->setParentWorld(this);
		registeredSystem->init();
		return registeredSystem;
	}

	template <typename ComponentType>
	void addComponent(Entity entity, const ComponentType& component) {
		componentManager->addComponent(entity, component);


		ComponentId componentId = getComponentId<ComponentType>();
		entityManager->setSignatureBit(entity, componentId, true);
		systemManager->entitySignatureChange(entity, entityManager->getSignature(entity));
	}

	template <typename ComponentType>
	ComponentType& getComponent(Entity entity) {
		return componentManager->getComponent<ComponentType>(entity);
	}

	template <typename ComponentType>
	void removeComponent(Entity entity) {
		componentManager->removeComponent<ComponentType>(entity);

		ComponentId componentId = getComponentId<ComponentType>();
		entityManager->setSignatureBit(entity, componentId, false);

		systemManager->entitySignatureChange(entity, entityManager->getSignature(entity));
	}

	void removeAllComponents(Entity entity) {
		// Do not destroy entity from entity manager since we only want to delete association with components
		componentManager->removeEntity(entity);
		systemManager->removeEntity(entity);
	}

	void update();
};