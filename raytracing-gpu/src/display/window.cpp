// #include <iostream>
// #include "window.h"


// Window::Window(const std::string& title, unsigned int width, unsigned int height) {
// 	this->dimensions.width = width;
// 	this->dimensions.height = height;

// 	// Initialize GLFW
// 	glfwInit();
// 	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
// 	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
// 	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

// 	id = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);

// 	// Check that GLFW was initiailized properly
// 	if (id == NULL) {
// 		glfwTerminate();
// 		std::cout << "Unable to create window" << std::endl;
// 	}

// 	glfwMakeContextCurrent(id);
// 	glfwSetInputMode(id, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

// 	// Set callbacks
// 	glfwSetWindowUserPointer(id, this);
// 	glfwSetKeyCallback(id, key_callback);
// 	glfwSetFramebufferSizeCallback(id, framebuffer_size_callback);
// 	glfwSetCursorPosCallback(id, cursor_pos_callback);
// 	glfwSetMouseButtonCallback(id, mouse_button_callback);

// 	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
// 		std::cout << "Unable to initialize GLAD" << std::endl;
// 	}

// 	glViewport(0, 0, width, height);
// 	glEnable(GL_DEPTH_TEST);
// 	glEnable(GL_STENCIL_TEST);

// }

// Window::~Window() {
// 	glfwTerminate();
// }

// void Window::clear() const {
// 	glClearColor(0.0, 0.2, 0.2, 1.0);
// 	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
// }

// void Window::update() const {
// 	glfwPollEvents();
// 	glfwSwapBuffers(id);
// }

// bool Window::isClosed()  const {
// 	return glfwWindowShouldClose(id);
// }

// /* Getters and setters */
// unsigned int Window::getWidth() const {
// 	return dimensions.width;
// }

// unsigned int Window::getHeight() const {
// 	return dimensions.height;
// }

// const Window::Input* Window::getInputs() const {
// 	return &inputs;
// }

// const Window::Dimensions* Window::getDimensions() const {
// 	return &dimensions;
// }
// // Callbacks


// void Window::framebuffer_size_callback(GLFWwindow* id, int width, int height) {
// 	Window* window = static_cast<Window*>(glfwGetWindowUserPointer(id));
// 	window->dimensions.height = height;
// 	window->dimensions.width = width;
// 	glViewport(0, 0, width, height);
// }

// void Window::cursor_pos_callback(GLFWwindow* id, double xPos, double yPos) {
// 	Window* window = static_cast<Window*>(glfwGetWindowUserPointer(id));
// 	Window::Input& inputs = window->inputs;

// 	inputs.mouseX = (float)xPos;
// 	inputs.mouseY = (float)yPos;
// }

// void Window::key_callback(GLFWwindow* id, int key, int scancode, int action, int mods) {
// 	Window* window = static_cast<Window*>(glfwGetWindowUserPointer(id));
// 	Window::Input& inputs = window->inputs;

// 	if (action == GLFW_PRESS) {
// 		inputs.pressedKeys[key] = true;
// 	} else if (action == GLFW_RELEASE) {
// 		inputs.pressedKeys[key] = false;
// 	}
// }

// void Window::mouse_button_callback(GLFWwindow* id, int button, int action, int mods) {
// 	Window* window = static_cast<Window*>(glfwGetWindowUserPointer(id));
// 	Window::Input& inputs = window->inputs;

// 	if (action == GLFW_PRESS) {
// 		inputs.pressedMouseButtons[button] = true;
// 	} else if (action == GLFW_RELEASE) {
// 		inputs.pressedMouseButtons[button] = false;
// 	}
// }