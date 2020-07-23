#pragma once
#include "../IMGUI/imgui.h"
#include "../IMGUI/imgui_impl_glfw.h"
#include "../IMGUI/imgui_impl_opengl3.h"
#include "SettingsWrapper.h"

#include <chrono>
#include <numeric>
#include <optional>


namespace GUI {
	struct {
		bool start{ false };
		bool stop{ false };
		bool pause{ false };
		bool resume{ false };
	} Signals;

	struct {
		bool mouseReleased{ false };
		bool mouseDown{ false };
		float dragX;
		float dragY;
		bool zoomOut;
		bool zoomIn;
	} Mouse;



	void initialize(GLFWwindow* window) {
		const char* glsl_version = "#version 460";
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		ImGui::StyleColorsDark();
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init(glsl_version);
	}

	static void statusText(bool status) {
		ImGui::SameLine();
		if (status) {
			ImGui::TextColored(ImVec4{ 0,1,0,1 }, "Valid");
		}
		else {
			ImGui::TextColored(ImVec4{ 1,0,0,1 }, "Invalid");
		}
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
			ImGui::Combo("Spawn Distribution", (int*) &(sw.Spawn.spawn_distr), "Uniform\0Gaussian\0Ring\0User defined P: R3 -> R");
			ImGui::Separator();
			if (sw.Spawn.spawn_distr == Spawn_Distr::USER_DEFINED) {
				static bool good = true;
				static char pdf[1024] = "x^2 + y^2 + z^2 < 100? 1:0";
				ImGui::InputText("P(x,y,z)", pdf, 1024); statusText(good);

				if (ImGui::TreeNode("Supported symbols")) {
					ImGui::Text("x     : First euclidean coordinate in right handed system.");
					ImGui::Text("y     : Second euclidean coordinate in right handed system.");
					ImGui::Text("z     : Third euclidean coordinate in right handed system.");
					ImGui::Text("r     : L2-norm on (x,y,z).");
					ImGui::Text("theta : Azimuth (rotation from x toward z).");
					ImGui::Text("phi   : Altitude (rotation from xz plane).");
					ImGui::TreePop();
				}
			}
			else {
				ImGui::InputFloat2("Parameters X", sw.Spawn.paramX.data(), 4);
				ImGui::InputFloat2("Parameters Y", sw.Spawn.paramY.data(), 4);
				ImGui::InputFloat2("Parameters Z", sw.Spawn.paramZ.data(), 4);
			}
			ImGui::Separator();
			ImGui::InputFloat2("Parameters mass", sw.Spawn.paramMass.data(), 4);
			ImGui::Checkbox("Black hole", &sw.Spawn.blackHole); ImGui::SameLine();
			ImGui::InputFloat("Mass factor", &sw.Spawn.blackHoleMassProportion, 0.02, 0.1, 4);
			ImGui::Combo("Angular momentum", (int*) &(sw.Spawn.angularMomentum), "None\0Inverse Magnitude\0Inverse magnitude squared\0Magnitude\0Magnitude Squared\0Uniform\0Gaussian\0User defined V: R3 -> R3");
			ImGui::Separator();
			if (sw.Spawn.angularMomentum == AngularMomentum_Distr::USER_DEFINED) {
				static bool goodX = true, goodY = true, goodZ = true;
				static char X[1024] = "r*sin(theta)";
				static char Y[1024] = "0";
				static char Z[1024] = "-r*cos(theta)";

				ImGui::InputText("V(x,y,z)[0]", X, 1024); statusText(goodX);
				ImGui::InputText("V(x,y,z)[1]", Y, 1024); statusText(goodY);
				ImGui::InputText("V(x,y,z)[2]", Z, 1024); statusText(goodZ);

				if (ImGui::TreeNode("Supported symbols")) {
					ImGui::Text("x     : First euclidean coordinate in right handed system.");
					ImGui::Text("y     : Second euclidean coordinate in right handed system.");
					ImGui::Text("z     : Third euclidean coordinate in right handed system.");
					ImGui::Text("r     : L2-norm on (x,y,z).");
					ImGui::Text("theta : Azimuth (rotation from x toward z).");
					ImGui::Text("phi   : Altitude (rotation from xz plane).");
					ImGui::TreePop();
				}
				
			}
			else {
				static bool recip = true;
				ImGui::Checkbox("", &recip); ImGui::SameLine();
				float amc = recip ? 1.0f / sw.Spawn.initialAngularMomentumCoefficent : sw.Spawn.initialAngularMomentumCoefficent;
				ImGui::DragFloat("Momentum factor", &(amc), recip ? 1.0f : 0.001f);
				sw.Spawn.initialAngularMomentumCoefficent = recip ? 1.0f / amc : amc;
			}
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
			Mouse.mouseReleased = true;
		}
		if (io.MouseDown[0]) {
			if (!ImGui::IsAnyWindowFocused()) {
				Mouse.dragX = io.MouseDelta.x;
				Mouse.dragY = io.MouseDelta.y;
				Mouse.mouseDown = true;
			}
			else {
				Mouse.dragX = 0;
				Mouse.dragY = 0;
				Mouse.mouseDown = false;
			}
		}

		if (io.MouseWheel > 0) {
			Mouse.zoomIn = true;
		}
		else if (io.MouseWheel < 0) {
			Mouse.zoomOut = true;
		}

		ImGui::End();
		ImGui::PopStyleVar(1);

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
};