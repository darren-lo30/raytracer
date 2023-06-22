#pragma once
#include "event_handler.h"
#include <vector>
#include <map>
#include <typeindex>
#include <vector>

typedef std::vector<EventHandlerBase*> HandlerList;

class EventManager {
private:
	std::map<std::type_index, HandlerList*> subscribers;
public:
	template <typename EventType>
	void publish(EventType* evnt) {
		HandlerList* subscribedHandlers = subscribers[typeid(EventType)];

		if (subscribedHandlers == nullptr) {
			return;
		}	

		for (auto& handler : *subscribedHandlers) {
			if (handler != nullptr) {
				handler->exec(evnt);
			}
		}
	}

	template <class T, class EventType>
	void subscribe(T* instance, void (T::* memberFunction)(EventType*)) {
		HandlerList* handlers = subscribers[typeid(EventType)];
		if (handlers == nullptr) {
			handlers = new HandlerList();
			subscribers[typeid(EventType)] = handlers;
		}

		handlers->push_back(new MemberEventHandler<T, EventType>(instance, memberFunction));
	}
};