#pragma once
#include <set>
#include <memory>
#include "entity.h"
#include "component.h"
#include "event_manager.h"

class World;
class EntityHandle;

class BaseSystem {
protected:
	Signature signature;
	World* parentWorld = nullptr;

	std::set<EntityHandle> entities;
public:
	BaseSystem() = default;

	EntityHandle wrapHandle(Entity entity);
	virtual void addEntity(Entity entity);
	virtual bool removeEntity(Entity entity);
	const Signature& getSignature() const;

	void setParentWorld(World* world);

	template <typename ComponentType>
	void registerComponent() {
		signature.set(getComponentId<ComponentType>());
	}

	virtual void init() = 0;	
};

class UpdateSystem : public BaseSystem {
protected:
	std::shared_ptr<EventManager> eventManager = nullptr;
public:
	void setEventManager(std::shared_ptr<EventManager> eventManager);
	virtual void update() = 0;
};

class RenderSystem : public BaseSystem {
public:
	virtual void render() = 0;
};