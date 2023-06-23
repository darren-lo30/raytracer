#pragma once

#include <cstdint>
#include <type_traits>
#include <iostream>

using ComponentId = std::uint8_t;

struct ComponentIdCounter {
	static int counter;
};


template <typename ComponentType>
struct Component {
	static inline int id() {
		static int id = ComponentIdCounter::counter++;
		return id;
	}
};

template <typename ComponentType>
static int getComponentId() {
	return Component<typename std::remove_const<ComponentType>::type>::id();
}

