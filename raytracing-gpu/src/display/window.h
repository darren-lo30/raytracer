#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string> 
#include <iostream>

class Window
{
public:
	static const int MAX_KEYS = 512;
	static const int MAX_MOUSE_BUTTONS = 64;

	struct Input {
		float mouseX, mouseY;
		bool pressedKeys[MAX_KEYS];
		bool pressedMouseButtons[MAX_MOUSE_BUTTONS];

		bool isPressed(GLenum key) const {
			return pressedKeys[key];
		}
	};

	struct Dimensions {
		unsigned int width;
		unsigned int height;
	};
private:
	GLFWwindow* id;
	Dimensions dimensions{};
	Input inputs{};
public:
	Window(const std::string& title, unsigned int width, unsigned int height);
	~Window();


	void clear() const;
	void update() const;
	bool isClosed() const;
	void close() const;

	GLFWwindow *getId() const;

	unsigned int getWidth() const ;
	unsigned int getHeight() const;

	const Input* getInputs() const;
	const Dimensions* getDimensions() const;
private:
	static void framebuffer_size_callback(GLFWwindow* id, int width, int height);
	static void key_callback(GLFWwindow* id, int key, int scancode, int action, int mods);
	static void cursor_pos_callback(GLFWwindow* id, double xPos, double yPos);
	static void mouse_button_callback(GLFWwindow* id, int button, int action, int mods);
};
