
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include "math_constants.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>

#include "kernel.cuh"
#include "SettingsWrapper.h"

#define USE_SHARED

#define cudaErrorCheck(...) { cudaErrorCheckingFunction((__VA_ARGS__), __FILE__, __LINE__); } 
#define blockSize ((int) 64)
#define blocksToLoad ((int) 2)

static Particle *plist;
static int numpart;
static Particle *device_plist;
static Vector3 *host_pos;

static inline void cudaErrorCheckingFunction(cudaError_t error, const char* file, int line, bool abort = true) {
	if (error != cudaSuccess) {
		fprintf(stderr, "Cuda error: %s %s %d\n", cudaGetErrorString(error), file, line);
		//throw std::runtime_error("CUDA error");
	}
}

static __device__ void setColour(Particle& p, int index, Vector3* outColor) {
	float magVel = magVector3(&p.v);
	//outColor[idx] = {fminf(1.0, 0.5/minDist) ,0.1, fminf(1.0f, minDist/0.5)};
	float r = fmaxf(fminf(magVel * 7, 1.0f), 0.1f);
	float g = fmaxf(fminf(magVel * 7 - 0.5, 1.0f), 0.1f);
	float b = fmaxf(fminf(magVel * 7 - 0.75, 1.0f), 0.1f);
	outColor[index] = { r, g, b };
}

static __global__ void gpu_doStepWithShared(Vector3 *outPos, Vector3 *outColor, Particle * nplist, int numP, float k0, float G, float timeStep) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	Vector3 newPos, newV;

	__shared__ Particle plist[blockSize*blocksToLoad];
	Particle self = nplist[idx];

	Vector3 force = { 0.0f,0.0f,0.0f };

	/*
	e.g., Block size = 8, sizeof(nplist) = 4*8 = 32, blocksToLoad = 2
	Blocks:	0        1        2        3
			|--------|--------|--------|--------|

	Each block will have  2 blocks loaded into shared memory.
	Block 1 threads will load on the second iteration:
	T0: nplist[16] and nplist[24]
	T1: 17 25
	T2: 18 26
	T3: 19 27
	T4: 20 28
	T5: 21 29
	T6: 22 30
	T7: 23 31

	T2 loads from:
	nplist:
	0        1        2        3
	|--------|--------|--X-----|--Y-----|
						 |        |
						 V        V
	into plist:       |--Z----- --W-----|
	*/

	//blockDim.x*blockSize;
	for (int block = 0; block < numP; block += blockSize * blocksToLoad) { //For every section of blocks to load into plist
		for (int i = 0; i < blocksToLoad; i++) {
			int offset = threadIdx.x + i * blockSize;
			plist[offset] = nplist[block + offset];
		}
		__syncthreads();
		for (int i = 0; i < blockSize*blocksToLoad; i++) {
			Vector3 r;
			copyVector3(&r, &self.pos);
			subVector3(&r, &plist[i].pos);
			float magR = magVector3(&r) + 0.05;

			float scaleFactor = -plist[i].m / (magR*magR*magR);

			Vector3 newForce = { 0.0f, 0.0f ,0.0f };
			copyVector3(&newForce, &r);
			scaleVector3(&newForce, G*scaleFactor);

			addVector3(&force, &newForce);
		}
	}

	
	newV.x = self.v.x + force.x*timeStep / self.m;
	
	newV.y = self.v.y + force.y*timeStep / self.m;
	
	newV.z = self.v.z + force.z*timeStep / self.m;

	float magNewV = magVector3(&newV);
	if (magNewV > 25) {
		normVector3(&newV);
		newV.x *= 25;
		newV.y *= 25;
		newV.z *= 25;
	}

	
	newPos.x = self.pos.x + timeStep * (self.v.x + newV.x)*0.5;
	newPos.y = self.pos.y + timeStep * (self.v.y + newV.y)*0.5;
	newPos.z = self.pos.z + timeStep * (self.v.z + newV.z)*0.5;

	if (self.isStationary) {
		copyVector3(&newPos, &self.pos);
		copyVector3(&newV, &self.v);
	}
	
	setColour(self, idx, outColor);
	copyVector3(&outPos[idx], &self.pos);
	copyVector3(&nplist[idx].pos, &newPos);
	copyVector3(&nplist[idx].v, &newV);
}

static __global__ void loadOutputs(Vector3 * pos, Vector3 * colour, Particle * nplist) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	setColour(nplist[idx], idx, colour);
	copyVector3(&pos[idx], &nplist[idx].pos);
}

void DeviceFunctions::loadData(Vector3 * pos, Vector3 * colour) {
	dim3 blocks(numpart / blockSize, 1, 1);
	dim3 threadsPerBlock(blockSize, 1, 1);

	loadOutputs <<< blocks, threadsPerBlock >>> (pos, colour, device_plist);
	cudaErrorCheck(cudaGetLastError());
	cudaDeviceSynchronize();
}

Particle * DeviceFunctions::getPlist() {
	return plist;
}

Vector3* DeviceFunctions::getParticlePos(Vector3 *pos) {
	cudaMemcpy(host_pos, pos, sizeof(Vector3)*numpart, cudaMemcpyDeviceToHost);
	return host_pos;
}

void DeviceFunctions::doStep(float timestep, Vector3 *pos, Vector3 *colour) {
	dim3 blocks(numpart / blockSize, 1, 1);
	dim3 threadsPerBlock(blockSize, 1, 1);

	cudaDeviceSynchronize();
	gpu_doStepWithShared <<< blocks, threadsPerBlock >>> (pos, colour, device_plist, numpart, 1.5, 0.0001, timestep);
	cudaErrorCheck(cudaGetLastError());
}

static float randFloat() {
	return (float)((double) rand() / (double)RAND_MAX);
}

static float randRange(float a, float b) {
	return ((float)rand() / RAND_MAX)*(b - a) + a;
}

static float randRange(std::array<float, 2> ab) {
	return ((float)rand() / RAND_MAX)*(ab[1] - ab[0]) + ab[0];
}

static float randGauss() {
	float u1 = randFloat(), u2 = randFloat();
	return sqrtf(-2*logf(u1))*cosf(2*CUDART_PI*u2);
}

static std::array<float,2> randGauss2() {

	float u1 = randFloat(), u2 = randFloat();
	return {
		sqrtf(-2*logf(u1))*cosf(2*CUDART_PI*u2),
		sqrtf(-2*logf(u1))*sinf(2*CUDART_PI*u2)
	};
}

float magnitude(const std::array<float, 3>& a) {
	return sqrtf(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}

void sampleFromSpawn(std::array<float,3>& z_prev, float& p_prev) {
	SettingsWrapper &sw = SettingsWrapper::get();
	const int samples = sw.Spawn.SpawnFunc_Samples;
	const float sigma = sw.Spawn.SpawnFunc_SigmaQ;
	for (int i = 0; i < samples; i++) {
		float p_prop;
		std::array<float, 3> z_prop;
		float u;
		do {
			z_prop = {
				sigma*randGauss() + z_prev[0],
				sigma*randGauss() + z_prev[1],
				sigma*randGauss() + z_prev[2] };
			p_prop = sw.Spawn.SpawnFunc(
				z_prop[0],
				z_prop[1],
				z_prop[2],
				magnitude(z_prop),
				atan2f(z_prop[2], z_prop[0]), //Azimuth
				atan2f(z_prop[1], z_prop[0]*z_prop[0] + z_prop[2]*z_prop[2]), //Altitude
				randFloat());
			u = randFloat();
		} while (abs(p_prop) / p_prev <= u);

		if (p_prev > p_prop) {
			printf("%f %f\n", p_prev, p_prop);
		}

		p_prev = abs(p_prop);
		z_prev = z_prop;
	}
}

void spawnParticles() {
	SettingsWrapper &sw = SettingsWrapper::get();
	const int sd = sw.Spawn.spawn_distr;

	std::array<float, 3> z_cur{ 0,0,0 };
	float p_cur;
	if (sd == Spawn_Distr::USER_DEFINED) {
		p_cur = sw.Spawn.SpawnFunc(
			z_cur[0],
			z_cur[1],
			z_cur[2],
			magnitude(z_cur),
			atan2f(z_cur[2], z_cur[0]), //Azimuth
			atan2f(z_cur[1], z_cur[0] * z_cur[0] + z_cur[2] * z_cur[2]), //Altitude
			randFloat());
	}


	for (int i = 0; i < numpart; i++) {
		plist[i].pos = { randRange(sw.Spawn.paramX),randRange(sw.Spawn.paramY),randRange(sw.Spawn.paramZ) };
		

		switch (sd) {
		case Spawn_Distr::GAUSS: {
			float x = randGauss()*sw.Spawn.paramX[1] + sw.Spawn.paramX[0];
			float y = randGauss()*sw.Spawn.paramY[1] + sw.Spawn.paramY[0];
			float z = randGauss()*sw.Spawn.paramZ[1] + sw.Spawn.paramZ[0];
			plist[i].pos = { x, y, z };
			break;
		}
		case Spawn_Distr::RING: {
			float theta1 = randRange(0, 2 * CUDART_PI);
			float r2 = randFloat()*sw.Spawn.paramY[1];
			float theta2 = randRange(0, 2 * CUDART_PI);
			float x = sw.Spawn.paramX[1] * cosf(theta1) + r2 * cosf(theta2)*cosf(theta1);
			float z = sw.Spawn.paramZ[1] * sinf(theta1) + r2 * cosf(theta2)*sinf(theta1);
			float y = r2 * sinf(theta2);
			plist[i].pos = { x, y, z };
			break;
		}
		case Spawn_Distr::UNIFORM: {
			plist[i].pos = { randRange(sw.Spawn.paramX),randRange(sw.Spawn.paramY),randRange(sw.Spawn.paramZ) };
			break;
		}
		case Spawn_Distr::USER_DEFINED: {
			sampleFromSpawn(z_cur, p_cur);
			copyVector3(&(plist[i].pos), z_cur);
			break;
		}
		default:
			plist[i].pos = { randRange(-5,5),randRange(-5,5),randRange(-5, 5) };
			break;
		}

		plist[i].v = { 0.0, 0.0, 0.0 };
		plist[i].m = randFloat() / numpart;
		plist[i].isStationary = 0;
	}
}

void initializeVelocity() {
	SettingsWrapper &sw = SettingsWrapper::get();

	for (int i = 0; i < numpart; i++) {
		float mag = magVector3(&plist[i].pos);
		Vector3 &pos = plist[i].pos;
		Vector3 ortho = { -pos.z, 0.0, pos.x };
		normVector3(&ortho);
		float scale;
		switch (sw.Spawn.angularMomentum) {
		case AngularMomentum_Distr::NONE:
			scale = 0;
			break;
		case AngularMomentum_Distr::MAG_SQ:
			scale = mag * mag;
			break;
		case AngularMomentum_Distr::MAG:
			scale = mag;
			break;
		case AngularMomentum_Distr::INV_MAG_SQ:
			scale = 1.0f / (mag*mag);
			break;
		case AngularMomentum_Distr::INV_MAG:
			scale = 1.0f / mag;
			break;
		case AngularMomentum_Distr::UNIFORM:
			scale = randFloat();
			break;
		case AngularMomentum_Distr::GAUSS:
			scale = 0;
			break;
		case AngularMomentum_Distr::USER_DEFINED:
			scale = 1;
			copyVector3(&ortho, sw.Spawn.VelocityFunc(
				pos.x,
				pos.y,
				pos.z,
				mag,
				atan2f(pos.z, pos.x), //Azimuth
				atan2f(pos.y, pos.x*pos.x + pos.z*pos.z), //Altitude
				randFloat()
			));
			break;
		default:
			scale = 0;
			break;

		}
		scaleVector3(&ortho, scale*sw.Spawn.initialAngularMomentumCoefficent);
		copyVector3(&plist[i].v, &ortho);
	}
}

int DeviceFunctions::setup(int num) {
	numpart = ((num - 1) / (blockSize*blocksToLoad) + 1)*blockSize*blocksToLoad;
	static bool firstRun = true;
	if(firstRun)
		cudaErrorCheck(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
	//printf("Startup...\n");
	//printf("Given %d particles, rounding to %d - the nearest multiple of blocksize*blocksToLoad (%d * %d)\n", num, numpart, blockSize, blocksToLoad);
	//printf("sizeof(Particle) = %d\n", sizeof(Particle));

	plist = (Particle*)malloc(sizeof(Particle)*numpart);
	host_pos = (Vector3*)malloc(sizeof(Vector3)*numpart);

	spawnParticles();
	initializeVelocity();

	cudaErrorCheck(cudaMalloc(&device_plist, sizeof(Particle)*numpart));
	cudaErrorCheck(cudaMemcpy(device_plist, plist, sizeof(Particle)*numpart, cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaDeviceSynchronize());
	firstRun = false;
	return numpart;
}

#undef blockSize
int DeviceFunctions::shutdown() {
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	
	cudaErrorCheck(cudaFree(device_plist));
	device_plist = nullptr;
	free(plist);
	free(host_pos);
	return 0;
}
