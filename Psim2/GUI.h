#pragma once
#include "../IMGUI/imgui.h"
#include "../IMGUI/imgui_impl_glfw.h"
#include "../IMGUI/imgui_impl_opengl3.h"
#include "SettingsWrapper.h"
#include "exprtk.hpp"

#include <chrono>
#include <numeric>


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

	static exprtk::symbol_table<float> symbol_table;
	static exprtk::expression<float> expression_pdf;
	static exprtk::expression<float> expression_velox;
	static exprtk::expression<float> expression_veloy;
	static exprtk::expression<float> expression_veloz;
	static exprtk::parser<float> parser;
	static struct {
		float x, y, z, r, theta, phi, u;
	} xpr;

	void initialize(GLFWwindow* window) {
		const char* glsl_version = "#version 460";
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		ImGui::StyleColorsDark();
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init(glsl_version);
		symbol_table.add_constants();
		symbol_table.add_variable("x", xpr.x);
		symbol_table.add_variable("y", xpr.y);
		symbol_table.add_variable("z", xpr.z);
		symbol_table.add_variable("r", xpr.r);
		symbol_table.add_variable("theta", xpr.theta);
		symbol_table.add_variable("phi", xpr.phi);
		symbol_table.add_variable("u", xpr.u);

		expression_pdf.register_symbol_table(symbol_table);
		expression_velox.register_symbol_table(symbol_table);
		expression_veloy.register_symbol_table(symbol_table);
		expression_veloz.register_symbol_table(symbol_table);

		SettingsWrapper& sw = SettingsWrapper::get();
		sw.Spawn.VelocityFunc = [&](
			float x,
			float y,
			float z,
			float r,
			float theta,
			float phi,
			float u) {
				symbol_table.get_variable("x")->ref() = x;
				symbol_table.get_variable("y")->ref() = y;
				symbol_table.get_variable("z")->ref() = z;
				symbol_table.get_variable("r")->ref() = r;
				symbol_table.get_variable("theta")->ref() = theta;
				symbol_table.get_variable("phi")->ref() = phi;
				symbol_table.get_variable("u")->ref() = u;

				return std::array<float, 3>{
					expression_velox.value(),
					expression_veloy.value(),
					expression_veloz.value(),
				};
		};

		sw.Spawn.SpawnFunc = [&](
			float x,
			float y,
			float z,
			float r,
			float theta,
			float phi,
			float u) {
				symbol_table.get_variable("x")->ref() = x;
				symbol_table.get_variable("y")->ref() = y;
				symbol_table.get_variable("z")->ref() = z;
				symbol_table.get_variable("r")->ref() = r;
				symbol_table.get_variable("theta")->ref() = theta;
				symbol_table.get_variable("phi")->ref() = phi;
				symbol_table.get_variable("u")->ref() = u;

				return expression_pdf.value();
		};
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

	static void fpsText() {
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

	void supportedSymbols() {
		if (ImGui::TreeNode("Supported symbols")) {
			ImGui::Text("x     : First euclidean coordinate in right handed system.");
			ImGui::Text("y     : Second euclidean coordinate in right handed system.");
			ImGui::Text("z     : Third euclidean coordinate in right handed system.");
			ImGui::Text("r     : L2-norm on (x,y,z).");
			ImGui::Text("theta : Azimuth (rotation from x toward z in radians).");
			ImGui::Text("phi   : Altitude (rotation from xz plane in radians).");
			ImGui::Text("u     : u~U(0,1).");
			ImGui::TreePop();
		}
	}

	void inputVelocityEquations() {
		SettingsWrapper& sw = SettingsWrapper::get();
		static bool first = true;
		static bool goodX = true, goodY = true, goodZ = true;
		static char X[1024] = "r*sin(theta)/15";
		static char Y[1024] = "0";
		static char Z[1024] = "-r*cos(theta)/15";

		if (ImGui::InputText("V(x,y,z)[0]", X, 1024) || first)
			goodX = parser.compile(X, expression_velox);
		statusText(goodX);
		if (ImGui::InputText("V(x,y,z)[1]", Y, 1024) || first)
			goodY = parser.compile(Y, expression_veloy);
		statusText(goodY);
		if (ImGui::InputText("V(x,y,z)[2]", Z, 1024) || first)
			goodZ = parser.compile(Z, expression_veloz);
		statusText(goodZ);

		sw.Spawn.VelocityFunc_good = goodX && goodY && goodZ;

		supportedSymbols();
		first = false;
	}

	void inputPDFEquation() {
		static bool first = false;
		SettingsWrapper& sw = SettingsWrapper::get();
		static char pdf[1024] = "x^2 + y^2 + z^2 < 100? 1:0";

		if (ImGui::InputText("P(x,y,z)", pdf, 1024) || first) {
			sw.Spawn.SpawnFunc_good = parser.compile(pdf, expression_pdf);
		}

		statusText(sw.Spawn.SpawnFunc_good);
	}

	void draw() {
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();


		static bool transparency = false;
		static bool recip = true;
		static float alpha = 1.0f;

		ImGui::PushStyleVar(ImGuiStyleVar_Alpha, transparency ? alpha : 1.0f);
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
			ImGui::InputFloat("Gravitational constant", &sw.SimulationFactors.gravConstant, 0.0f, 0.0f, "%.5e");
			ImGui::InputFloat("Time step", &sw.SimulationFactors.timeStep, 0.0f, 0.0f, "%.5e");
		}

		if (ImGui::CollapsingHeader("Initial parameters")) {
			ImGui::InputScalar("N", ImGuiDataType_U64, (void*)&sw.Spawn.N);
			ImGui::Combo("Spawn Distribution", (int*) &(sw.Spawn.spawn_distr), "Uniform\0Gaussian\0Ring\0User defined P: R3 -> R");
			ImGui::Separator();
			if (sw.Spawn.spawn_distr == Spawn_Distr::USER_DEFINED) {
				
				inputPDFEquation();

				supportedSymbols();
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
				inputVelocityEquations();
			}
			else {
				
				ImGui::Checkbox("", &recip);
				if (ImGui::IsItemClicked())
					recip = !recip;

				ImGui::SameLine();
				float amc = recip ? 1.0f / sw.Spawn.initialAngularMomentumCoefficent : sw.Spawn.initialAngularMomentumCoefficent;
				ImGui::DragFloat("Momentum factor", &(amc), recip ? 1.0f : 0.001f);
				sw.Spawn.initialAngularMomentumCoefficent = recip ? 1.0f / amc : amc;
			}

			ImGui::Separator();
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