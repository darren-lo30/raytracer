// #include "BasicRenderSystem.h"
// #include "CameraComponent.h"
// #include "Renderer.h"
// #include "View.h"


// BasicRenderSystem::BasicRenderSystem(const EntityHandle* camera) {
// 	this->camera = camera;
// }

// void BasicRenderSystem::init() {
// 	registerComponent<Model>();
// 	registerComponent<Transform>();

// 	basicShader = std::make_unique<Shader>(Shader("shaders/basicShader.vert", "shaders/basicShader.frag"));
// 	basicShader->link();
// }

// float vertices[] = {
// 	// first triangle
// 	 0.5f,  0.5f, 0.0f,  // top right
// 	 0.5f, -0.5f, 0.0f,  // bottom right
// 	-0.5f,  0.5f, 0.0f,  // top left 
// };

// unsigned int VAO = 0;

// void BasicRenderSystem::render() {
// 	unsigned int VBO;
// 	if (!VAO) {
// 		glGenVertexArrays(1, &VAO);
// 		glBindVertexArray(VAO);
		
// 		glGenBuffers(1, &VBO);
// 		glBindBuffer(GL_ARRAY_BUFFER, VBO);
// 		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

// 		glEnableVertexAttribArray(0);
// 		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

// 		glBindVertexArray(0);
// 	}

// 	basicShader->use();
// 	for (const auto& entity : entities) {
// 		auto& transform = entity.getComponent<Transform>();
// 		auto& model = entity.getComponent<Model>();

// 		glm::mat4 modelMat = glm::mat4(1.0);
// 		modelMat = glm::translate(modelMat, transform.position);

// 		modelMat *= glm::mat4_cast(transform.rotation);
// 		modelMat = glm::scale(modelMat, transform.scale);

// 		basicShader->setMat4("model", modelMat);

// 		basicShader->setMat4("view", getViewMat());
// 		basicShader->setMat4("projection", getProjectionMat());

// 		Renderer::renderModel(model, *basicShader);
// 	}
// }

// glm::mat4 BasicRenderSystem::getViewMat() {
// 	auto& cameraTransform = camera->getComponent<Transform>();
// 	auto cameraDirections = cameraTransform.getDirectionVectors();

// 	return glm::lookAt(cameraTransform.position, cameraTransform.position + cameraDirections.front, glm::vec3(0, 1, 0));
// }

// glm::mat4 BasicRenderSystem::getProjectionMat() {
// 	auto& cameraComponent = camera->getComponent<CameraComponent>();
// 	return glm::perspective(
// 		cameraComponent.FOV,
// 		(float) cameraComponent.viewport->width / cameraComponent.viewport->height,
// 		cameraComponent.nearPlane,
// 		cameraComponent.farPlane
// 	);
// }




