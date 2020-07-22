#pragma once
#include "../IMGUI/imgui.h"
#include "../IMGUI/imgui_impl_glfw.h"
#include "../IMGUI/imgui_impl_opengl3.h"
#include "SettingsWrapper.h"

#include <chrono>
#include <numeric>


namespace GUI {
	struct {
		bool start{ false };
		bool stop{ false };
		bool pause{ false };
		bool resume{ false };
		bool mouseReleased{ false };
		bool mouseDown{ false };
		float dragX;
		float dragY;
		bool zoomOut;
		bool zoomIn;
	} Signals;



	void initialize(GLFWwindow* window) {
		const char* glsl_version = "#version 460";
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		ImGui::StyleColorsDark();
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init(glsl_version);
	}

	static void fpsText(){
		static std::array<float, 100> frameTimes;
		static int insertIndex = 0;

		static auto lastTime = std::chrono::system_clock::now();
		auto nowTime = std::chrono::system_clock::now();
		float delta = std::chrono::duration_cast<std::chrono::nanoseconds>(
			(std::chrono::duration<float>)(nowTime - lastTime)).count() / 1E9f;

		lastTime = nowTime;

		frameTimes[insertIndex] = delta;
		insertIndex = (insertIndex + 1) % 100;

		float avgFrameTime = std::accumulate(frameTimes.begin(), frameTimes.end(), 0.0f) / 100.0f;

		ImGui::Text("%.0f FPS", 1.0f / avgFrameTime);
	}

	void draw() {
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();


		static bool transparency = false;
		static float alpha = 1.0f;
		
		ImGui::PushStyleVar(ImGuiStyleVar_Alpha, transparency ? alpha:1.0f);
		ImGui::Begin("Settings");

		ImGui::Text("Window settings");
		ImGui::Checkbox("", &transparency);
		ImGui::SameLine();
		ImGui::SliderFloat("Transparency", &alpha, 0, 1);

		fpsText();

		ImGui::Separator();
		ImGui::Text("Simulation settings");

		SettingsWrapper& sw = SettingsWrapper::get();

		float width = ImGui::GetColumnWidth();
		float height = ImGui::GetTextLineHeightWithSpacing();
		ImGui::Button("Start", ImVec2(width / 4, height));
		if (ImGui::IsItemClicked()) {
			Signals.start = true;
		}
		ImGui::SameLine();
		ImGui::Button("Stop", ImVec2(width / 4, height));
		if (ImGui::IsItemClicked()) {
			Signals.stop = true;
		}
		ImGui::SameLine();
		ImGui::Button("Resume", ImVec2(width / 4, height));
		if (ImGui::IsItemClicked()) {
			Signals.resume = true;
		}

		ImGui::SameLine();
		ImGui::Button("Pause", ImVec2(width / 4, height));
		if (ImGui::IsItemClicked()) {
			Signals.pause = true;
		}
		

		if (ImGui::CollapsingHeader("Simulation factors")) {
			ImGui::InputFloat("Gravitational constant", &sw.SimulationFactors.gravConstant,0.0f, 0.0f, "%.5e");
			ImGui::InputFloat("Time step", &sw.SimulationFactors.timeStep, 0.0f, 0.0f, "%.5e");
		}

		if (ImGui::CollapsingHeader("Initial parameters")) {
			ImGui::InputScalar("N", ImGuiDataType_U64, (void*)&sw.Spawn.N);
			ImGui::Combo("Spawn Distribution", (int*) &(sw.Spawn.spawn_distr), "Uniform\0Gaussian\0Ring");
			ImGui::InputFloat2("Parameters X", sw.Spawn.paramX.data(), 4);
			ImGui::InputFloat2("Parameters Y", sw.Spawn.paramY.data(), 4);
			ImGui::InputFloat2("Parameters Z", sw.Spawn.paramZ.data(), 4);
			ImGui::InputFloat2("Parameters mass", sw.Spawn.paramMass.data(), 4);
			ImGui::Checkbox("Black hole", &sw.Spawn.blackHole); ImGui::SameLine();
			ImGui::InputFloat("Mass factor", &sw.Spawn.blackHoleMassProportion, 0.02, 0.1, 4);
			ImGui::Combo("Angular momentum", (int*) &(sw.Spawn.angularMomentum), "None\0Inverse Magnitude\0Inverse magnitude squared\0Magnitude\0Magnitude Squared\0Uniform\0Gaussian");
			static bool recip = true;
			ImGui::Checkbox("Reciprocal", &recip); ImGui::SameLine();
			float amc = recip? 1.0f/sw.Spawn.initialAngularMomentumCoefficent:sw.Spawn.initialAngularMomentumCoefficent;
			ImGui::DragFloat("Momentum factor", &(amc), recip? 1.0f:0.001f);
			sw.Spawn.initialAngularMomentumCoefficent = recip ? 1.0f/amc:amc;

		}


		if (ImGui::CollapsingHeader("Camera parameters")) {
			ImGui::SliderFloat("Azimuth", &(sw.Camera.azimuth), -180, 180, "%.1f degrees");
			ImGui::SliderFloat("Altitude", &(sw.Camera.altitude), -90, 90, "%.1f degrees");
			ImGui::SliderFloat("Radius", &(sw.Camera.radius), 0.1f, 1000.0f, "%.3f", 3.0f);
			ImGui::SliderFloat("Fov", &(sw.Camera.fov), 1.0f, 120.0f);
		}

		sw.enforceBounds();
		ImGuiIO& io = ImGui::GetIO();
		if (io.MouseReleased[0]) {
			Signals.mouseReleased = true;
		}
		if (io.MouseDown[0]) {
			if (!ImGui::IsAnyWindowFocused()) {
				Signals.dragX = io.MouseDelta.x;
				Signals.dragY = io.MouseDelta.y;
				Signals.mouseDown = true;
			}
			else {
				Signals.dragX = 0;
				Signals.dragY = 0;
				Signals.mouseDown = false;
			}
		}

		if (io.MouseWheel > 0) {
			Signals.zoomIn = true;
		}
		else if (io.MouseWheel < 0) {
			Signals.zoomOut = true;
		}

		ImGui::End();
		ImGui::PopStyleVar(1);

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
};