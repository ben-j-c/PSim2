#pragma once

#include "Vector3.cuh"

typedef struct Particle {
	Vector3 pos;
	Vector3 v;
	float q, m;
	bool isStationary;
} Particle;


namespace DeviceFunctions {
	Particle* getPlist();
	Vector3* getParticlePos(Vector3 *pos);
	int setup(int num);
	void doStep(float timeStep, Vector3 *pos, Vector3* colour);
	void loadData(Vector3 *pos, Vector3* colour);
	int shutdown();
}
