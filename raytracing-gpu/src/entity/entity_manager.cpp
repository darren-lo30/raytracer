#include "entity_manager.h"

EntityManager::EntityManager() {
	numLivingEntities = 0;

	for (Entity e = 1; e < MAX_ENTITIES; e++) {
		availableEntities.push(e);
	}
}

Entity EntityManager::createEntity() {
	// Get next available entity from queue
	Entity entity = availableEntities.front();
	availableEntities.pop();

	entitySignatures[entity].reset();

	numLivingEntities++;

	return entity;
}

void EntityManager::destroyEntity(Entity entity) {
	// Make entity id available again
	availableEntities.push(entity);

	numLivingEntities--;
}

void EntityManager::setSignature(Entity entity, const Signature& signature) {
	entitySignatures[entity] = signature;
}


void EntityManager::setSignatureBit(Entity entity, int bit, bool val) {
	Signature& signature = entitySignatures[entity];
	signature.set(bit, val);
}

const Signature& EntityManager::getSignature(Entity entity) const {
	return entitySignatures[entity];	
}
