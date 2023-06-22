#pragma once

#include "event.h"

class EventHandlerBase {
public:
	void exec(Event* event) {
		call(event);
	}
private:
	virtual void call(Event* event) = 0;
};

template <class T, class EventType>
class MemberEventHandler : public EventHandlerBase {
public:
	typedef void (T::*MemberFunction)(EventType*);

	MemberEventHandler(T* instance, MemberFunction memberFunction) : instance{ instance }, memberFunction{ memberFunction } {};

	void call(Event* event) {
		(instance->*memberFunction)(static_cast<EventType*>(event));
	}

private:
	T* instance;
	MemberFunction memberFunction;
};