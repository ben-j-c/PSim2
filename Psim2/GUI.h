#pragma once
#include "../IMGUI/imgui.h"
#include "../IMGUI/imgui_impl_glfw.h"
#include "../IMGUI/imgui_impl_opengl3.h"


namespace GUI {

	void initialize(GLFWwindow* window) {
		const char* glsl_version = "#version 460";
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		ImGui::StyleColorsDark();
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init(glsl_version);
	}

	void draw() {
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("Settings");


		if (ImGui::CollapsingHeader("Simulation parameters")) {
			;
		}


		static struct {
			float azimuth, altitude, radius = 10.0f, span = 2.0f;
		} Camera;
		if (ImGui::CollapsingHeader("Camera parameters")) {
			ImGui::SliderFloat("Azimuth", &(Camera.azimuth), -180, 180, "%.1f degrees");
			ImGui::SliderFloat("Altitude", &(Camera.altitude), -90, 90, "%.1f degrees");
			ImGui::SliderFloat("Radius", &(Camera.radius), 0.1f, 1000.0f, "%.3f", 3.0f);
			ImGui::SliderFloat("Span", &(Camera.span), 0.1f, 10.0f);
		}

		ImGui::End();


		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
};