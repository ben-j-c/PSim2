#include <stdio.h>
#include <memory>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "GUI.h"
#include "graphics/Graphics.h"

int display_w, display_h;

static void glfw_error_callback(int error, const char* description) {
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int main() {
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		return 1;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	GLFWwindow* window = glfwCreateWindow(640 * 2, 720,
		"PSim2", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open window\n");
		return 1;
	}
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	if (glewInit()) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return 1;
	}

	GUI::initialize(window);


	Graphics::VAObject VAO;
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glfwGetFramebufferSize(window, &display_w, &display_h);

		glBindVertexArray(*VAO); checkGL("Bind VAO");

		glViewport(0, 0, display_w, display_h);
		glClearColor(0.25f, 0.25f, 0.25f, 0);
		glClear(GL_COLOR_BUFFER_BIT);

		GUI::draw();

		glfwSwapBuffers(window);
	}

	return 0;
}