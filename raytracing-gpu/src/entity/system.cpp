#include "system.h"
#include "entity_handle.h"

#include <iostream>

EntityHandle BaseSystem::wrapHandle(Entity entity) {
	return EntityHandle(entity, parentWorld);
}

void BaseSystem::addEntity(Entity entity) {
	entities.insert(wrapHandle(entity));
}

bool BaseSystem::removeEntity(Entity entity) {
	auto it = entities.find(EntityHandle(entity, NULL));
	if (it == entities.end()) {
		return false;
	}

	entities.erase(it);
	return true;
}

const Signature& BaseSystem::getSignature() const {
	return signature;
}


void BaseSystem::setParentWorld(World* parentWorld) {
	this->parentWorld = parentWorld;
}

void UpdateSystem::setEventManager(std::shared_ptr<EventManager> eventManager) {
	this->eventManager = eventManager;
}

