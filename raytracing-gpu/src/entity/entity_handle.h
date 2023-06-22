#pragma once

#include "entity.h"
#include "world.h"

class EntityHandle {
private:
	Entity id;
	World* parentWorld;
public: 
	EntityHandle() {
		id = NULL;
		parentWorld = NULL;
	}

	EntityHandle(Entity id, World* parentWorld) {
		this->id = id;
		this->parentWorld = parentWorld;
	}

	template <typename ComponentType>
	void addComponent(const ComponentType& component) const {
		parentWorld->addComponent(id, component);
	}

	template <typename ComponentType>
	void removeComponent() const {
		parentWorld->removeComponent<ComponentType>(id);
	}

	template <typename ComponentType>
	ComponentType& getComponent() const {
		return parentWorld->getComponent<ComponentType>(id);
	}

	Entity getId() const {
		return id;
	}

	bool operator<(const EntityHandle& other) const {
		return id < other.id;
	}
};