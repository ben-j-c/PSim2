#include "Model.h"
#include "kernel.cuh"
#include "SettingsWrapper.h"

Model::Model(size_t N, GLuint program) : cloud(DeviceFunctions::setup(N)), program(program) {
}

Model::~Model() {
	DeviceFunctions::shutdown();
}

void Model::step() {
	SettingsWrapper& sw = SettingsWrapper::get();
	DeviceFunctions::doStep(sw.SimulationFactors.timeStep, (Vector3*) cloud.vPos_device, (Vector3*) cloud.vColour_device); //TODO: replace Vector3 with vec3
	auto a = cudaGetLastError();
}

void Model::draw() {
	cloud.draw(program);
}

void Model::releaseCuda() {
	cudaDeviceReset();
}
