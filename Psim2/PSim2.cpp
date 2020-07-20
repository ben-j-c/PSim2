#define DEFAULT_PROG "prog_point_cloud"
#define SHADER_VERT "./shaders/vertex.glsl"
#define SHADER_FRAG "./shaders/fragment.glsl"


#include <stdio.h>
#include <memory>
#define _USE_MATH_DEFINES
#include <math.h>
#include <optional>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "GUI.h"
#include "graphics/Graphics.h"
#include "graphics/GraphicsState.hpp"
#include "graphics/ShaderHandler.hpp"
#include "Model.h"

int display_w, display_h;

static void glfw_error_callback(int error, const char* description) {
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

static void poseCamera() {
	SettingsWrapper& sw = SettingsWrapper::get();
	float ratio = (float)display_w / display_h;
	float r = sw.Camera.radius;
	float alt = sw.Camera.altitude*M_PI/180.0f;
	float azi = sw.Camera.azimuth*M_PI/180.0f;
	Graphics::vec4 camPos = {
		r*cosf(azi)*cosf(alt),
		r*sinf(alt),
		r*sinf(azi)*cosf(alt),
		0.0f
	};
	//Graphics::setCamera2D(ratio, 10, { 0,0,0,0 });
	Graphics::setCamera(sw.Camera.fov, ratio, 0.1f, 1000.0f, camPos, { 0,0,0,0 }, { 0,1,0,0 });
}

int main() {
	try {

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
		glfwSwapInterval(0);

		if (glewInit()) {
			fprintf(stderr, "Failed to initialize GLEW\n");
			return 1;
		}

		GUI::initialize(window);
		glfwSetErrorCallback(glfw_error_callback);
		glDebugMessageCallback(Graphics::MessageCallback, nullptr);

		Graphics::VAObject VAO;
		glBindVertexArray(*VAO); checkGL("Bind VAO");
		ShaderHandler::createProgram(DEFAULT_PROG);
		ShaderHandler::loadVertexShader(DEFAULT_PROG, SHADER_VERT);
		ShaderHandler::loadFragmentShader(DEFAULT_PROG, SHADER_FRAG);
		ShaderHandler::attachShader(DEFAULT_PROG, SHADER_VERT);
		ShaderHandler::attachShader(DEFAULT_PROG, SHADER_FRAG);
		ShaderHandler::linkShaders(DEFAULT_PROG);

		printf("%s\n", glGetString(GL_VERSION));

		SettingsWrapper& sw = SettingsWrapper::get();
		bool stepSim = false, drawSim = true;
		std::optional<Model> model;
		model.emplace(sw.Spawn.N, ShaderHandler::getProgram(DEFAULT_PROG));
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			glfwGetFramebufferSize(window, &display_w, &display_h);

			ShaderHandler::switchProgram(DEFAULT_PROG);
			glBindVertexArray(*VAO); checkGL("Bind VAO");

			glEnable(GL_BLEND); checkGL("blend enable");
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); checkGL("blend func set");
			glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ZERO); checkGL("blend func seperate");
			glPointSize(2.0f); checkGL("point size");
			glViewport(0, 0, display_w, display_h);
			glClearColor(0.25f, 0.25f, 0.25f, 0);
			glClear(GL_COLOR_BUFFER_BIT);
			
			poseCamera();

			
			if (GUI::Signals.start) {
				if (!model) {
					model.emplace(sw.Spawn.N, ShaderHandler::getProgram(DEFAULT_PROG));
					stepSim = false;
					drawSim = true;
				}
			}
			else if (GUI::Signals.stop) {
				model.reset();
				stepSim = false;
				drawSim = false;
			}
			else if (GUI::Signals.resume) {
				if (model) {
					stepSim = true;
				}
			}
			else if (GUI::Signals.pause) {
				if (model) {
					stepSim = false;
				}
			}

			if (drawSim)
				model->draw();
			checkGL("idk");
			if (stepSim)
				model->step();

			GUI::Signals = { false, false, false, false };

			GUI::draw();

			glfwSwapBuffers(window);
		}

		Model::releaseCuda();
		ShaderHandler::shutdownShaders(DEFAULT_PROG);
	}
	catch (std::exception e) {
		std::fprintf(stderr, "%s\n", e.what());
	}
	return 0;
}