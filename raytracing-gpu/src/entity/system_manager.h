#pragma once

#include <memory>
#include <unordered_map>

#include "entity.h"
#include "system.h"

class SystemManager {
private:
	std::unordered_map<const char*, std::shared_ptr<UpdateSystem>> updateSystems;
	std::unordered_map<const char*, std::shared_ptr<RenderSystem>> renderSystems;
public:

	template <typename SystemType, typename... Args>
	std::shared_ptr<SystemType> registerUpdateSystem(Args&& ... args) {
		const char* systemTypeName = typeid(SystemType).name();
		assert(updateSystems.find(systemTypeName) == updateSystems.end() && "System is already registred");

		auto registeredSystem = std::make_shared<SystemType>(std::forward<Args>(args)...);
		updateSystems.insert({ systemTypeName, registeredSystem });
		return registeredSystem;
	}

	template <typename SystemType, typename... Args>
	std::shared_ptr<SystemType> registerRenderSystem(Args&& ... args) {
		const char* systemTypeName = typeid(SystemType).name();
		assert(renderSystems.find(systemTypeName) == renderSystems.end() && "System is already registred");

		auto registeredSystem = std::make_shared<SystemType>(std::forward<Args>(args)...);
		renderSystems.insert({ systemTypeName, registeredSystem });
		return registeredSystem;
	}

	void removeEntity(Entity entity) {
		for (auto const& p : updateSystems) {
			p.second->removeEntity(entity);
		}
		for (auto const& p : renderSystems) {
			p.second->removeEntity(entity);
		}
	}

	void entitySignatureChange(Entity entity, Signature entitySignature) {
		for (auto const& p : updateSystems) {
			auto const& system = p.second;

			if ((entitySignature & system->getSignature()) == system->getSignature()) {
				system->addEntity(entity);
			} else {
				system->removeEntity(entity);
			}
		}

		for (auto const& p : renderSystems) {
			auto const& system = p.second;

			if ((entitySignature & system->getSignature()) == system->getSignature()) {
				system->addEntity(entity);
			} else {
				system->removeEntity(entity);
			}
		}
	}

	void update() {
		for (auto const& p : updateSystems) {
			auto& system = p.second;
			system->update();
		}
	}

	void render() const {
		for (auto const& p : renderSystems) {
			auto& system = p.second;
			system->render();
		}
	}
};

