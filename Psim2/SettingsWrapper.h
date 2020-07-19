#pragma once
#include <memory>
#include <array>


namespace Spawn_Distr{
	typedef enum Spawn_Distr {
		UNIFORM = 0, GAUSS = 1, RING = 2
	};
};

namespace  AngularMomentum_Distr {
	typedef enum AngularMomentum_Distr {
		NONE = 0, INV_MAG = 1, INV_MAG_SQ = 2, MAG = 3, MAG_SQ = 4, UNIFORM = 5, GAUSS = 6
	};
};

class SettingsWrapper {
private:
	SettingsWrapper() {};

	SettingsWrapper(SettingsWrapper&) = delete;
	SettingsWrapper(SettingsWrapper&&) = delete;
	SettingsWrapper(const SettingsWrapper&) = delete;

	static std::unique_ptr<SettingsWrapper> singleton;

	template<class T>
	static void clamp(T& val, T min, T max) {
		if (val < min)
			val = min;
		if (val > max)
			val = max;
	}


public:
	struct {
		float azimuth = 0.0f, altitude = 0.0f, radius = 10.0f, fov = 80.0f;
	} Camera;

	struct {
		size_t N = 10000;
		std::array<float, 2>
			paramX{ -5, 5 },
			paramY{ -1, 1 },
			paramZ{ -5,5 },
			paramMass{0, 1.0f};
		Spawn_Distr::Spawn_Distr spawn_distr = Spawn_Distr::UNIFORM;
		AngularMomentum_Distr::AngularMomentum_Distr angularMomentum = AngularMomentum_Distr::INV_MAG;
		float initialAngularMomentumCoefficent = 1.0f/250.0f;

		bool blackHole = false;
		float blackHoleMassProportion = 0.25f;
		
	} Spawn;

	struct {
		float timeStep = 1.0f;
		float gravConstant = 0.0001f;
	} SimulationFactors;


	static SettingsWrapper& get() {
		return *singleton;
	};

	void enforceBounds() {
		clamp(Camera.azimuth, -180.0f, 180.0f);
		clamp(Camera.altitude, -90.0f, 90.0f);
		clamp(Camera.radius, 0.01f, INFINITY);
		clamp(Camera.fov, 0.1f, 120.0f);

		clamp(Spawn.N, 0ui64, SIZE_MAX);
		clamp(Spawn.blackHoleMassProportion, 0.0f, INFINITY);
		clamp(SimulationFactors.gravConstant, 0.0f, INFINITY);
		clamp(SimulationFactors.timeStep, 0.0f, INFINITY);
	}
};
