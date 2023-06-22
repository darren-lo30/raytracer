#pragma once

#include <queue>
#include <array>

#include "entity.h"

class EntityManager {
private:
	std::queue<Entity> availableEntities;
	std::array<Signature, MAX_ENTITIES> entitySignatures;
	unsigned int numLivingEntities;
public:
	EntityManager();
	Entity createEntity();
	void destroyEntity(Entity entity);
	void setSignature(Entity entity, const Signature& signature);
	void setSignatureBit(Entity entity, int bit, bool val);
	const Signature& getSignature(Entity entity) const;
};
