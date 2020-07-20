
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "kernel.cuh"

#define USE_SHARED

#define cudaErrorCheck(...) { cudaErrorCheckingFunction((__VA_ARGS__), __FILE__, __LINE__); } 
#ifdef USE_SHARED
#define blockSize ((int) 8)
#define blocksToLoad ((int) 32)
#define ifUseShared(...) {__VA_ARGS__;}
#define ifNotUseShared(a) {;}
#else
#define blockSize ((int) 32)
#define blocksToLoad ((int) 1)
#define ifUseShared(a) {;}
#define ifNotUseShared(...) {__VA_ARGS__;}
#endif

static Particle *plist;
static int numpart;
static Particle *device_plist;
static Vector3 *host_pos;

static inline void cudaErrorCheckingFunction(cudaError_t error, const char* file, int line, bool abort = true) {
	if (error != cudaSuccess) {
		fprintf(stderr, "Cuda error: %s %s %d\n", cudaGetErrorString(error), file, line);
		if (abort) exit(error);
	}
}

static __global__ void gpu_doStep(Vector3 *outPos, Vector3 *outColor, Particle * nplist, int numP, float k0, float G, float timeStep) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	Vector3 newPos, newV;

	Particle self = nplist[idx];

	if (!self.isStationary) {
		Vector3 force = { 0.0f,0.0f,0.0f };

		for (int i = 0; i < numP; i++) {
			if (i != idx) {
				Vector3 r;
				copyVector3(&r, &self.pos);
				subVector3(&r, &nplist[i].pos);
				float magR = magVector3(&r) + 0.05;

				float scaleFactor = (-nplist[i].m) / (magR*magR*magR);

				Vector3 newForce = { 0.0f, 0.0f ,0.0f };
				copyVector3(&newForce, &r);
				scaleVector3(&newForce, G*scaleFactor);

				addVector3(&force, &newForce);
			}
		}

		newV.x = self.v.x + force.x*timeStep / self.m;
		newV.y = self.v.y + force.y*timeStep / self.m;
		newV.z = self.v.z + force.z*timeStep / self.m;

		newPos.x = self.pos.x + timeStep * (self.v.x + newV.x)*0.5;
		newPos.y = self.pos.y + timeStep * (self.v.y + newV.y)*0.5;
		newPos.z = self.pos.z + timeStep * (self.v.z + newV.z)*0.5;
	}
	else {
		copyVector3(&newPos, &self.pos);
		copyVector3(&newV, &self.v);
	}

	float magVel = magVector3(&self.v);
	float r = fmaxf(fminf(magVel * 15, 1.0f), 0.1f);
	float g = fmaxf(fminf(magVel * 15 - 0.5, 1.0f), 0.1f);
	float b = fmaxf(fminf(magVel * 15 - 0.75, 1.0f), 0.1f);
	outColor[idx] = { r, g, b };

	copyVector3(&outPos[idx], &newPos);
	copyVector3(&nplist[idx].pos, &newPos);
	copyVector3(&nplist[idx].v, &newV);
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
	for (int block = 0; block < blockDim.x*blockSize; block += blockSize * blocksToLoad) { //For every section of blocks to load into plist
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

	float magVel = magVector3(&self.v);
	//outColor[idx] = {fminf(1.0, 0.5/minDist) ,0.1, fminf(1.0f, minDist/0.5)};
	float r = fmaxf(fminf(magVel * 15, 1.0f), 0.1f);
	float g = fmaxf(fminf(magVel * 15 - 0.5, 1.0f), 0.1f);
	float b = fmaxf(fminf(magVel * 15 - 0.75, 1.0f), 0.1f);
	outColor[idx] = { r, g, b };
	copyVector3(&outPos[idx], &newPos);
	copyVector3(&nplist[idx].pos, &newPos);
	copyVector3(&nplist[idx].v, &newV);
}

static __global__ void computeVelocity(Particle* parts) {
	int i = threadIdx.x + blockDim.x*blockIdx.x;

	float mag = magVector3(&parts[i].pos);
	Vector3 ortho = { -parts[i].pos.z, 0.0, parts[i].pos.x };
	normVector3(&ortho);
	scaleVector3(&ortho, mag / 250.0);
	copyVector3(&parts[i].v, &ortho);
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

#if defined(USE_SHARED)
	gpu_doStepWithShared <<< blocks, threadsPerBlock >>> (pos, colour, device_plist, numpart, 1.5, 0.0001, (float)timestep);
	cudaDeviceSynchronize();
#else
	gpu_doStep << < blocks, threadsPerBlock >> > (pos, colour, device_plist, numpart, 1.5, 0.0001, (float)timestep)
#endif
}

static float randFloat() {
	return (float)rand() / RAND_MAX;
}

static float randRange(float a, float b) {
	return ((float)rand() / RAND_MAX)*(b - a) + a;
}

int DeviceFunctions::setup(int num) {
	numpart = ((num - 1) / (blockSize*blocksToLoad) + 1)*blockSize*blocksToLoad;
	printf("Startup...\n");
	printf("Given %d particles, rounding to %d - the nearest multiple of blocksize*blocksToLoad (%d * %d)\n", num, numpart, blockSize, blocksToLoad);
	ifUseShared(printf("Using shared memory\n"));
	ifNotUseShared(printf("Not using shared memory\n"));
	printf("sizeof(Particle) = %d\n", sizeof(Particle));
	plist = (Particle*)malloc(sizeof(Particle)*numpart);
	host_pos = (Vector3*)malloc(sizeof(Vector3)*numpart);


	for (int i = 0; i < numpart; i++) {
		plist[i].pos = { randRange(-10,10),randRange(-5,5),randRange(-10,10) };
		plist[i].v = { 0.0, 0.0, 0.0 };
		plist[i].m = randFloat() / numpart;
		plist[i].q = randRange(0.001, 0.001) / numpart; //10% should be weakly negative. This should allow for clumping
		plist[i].isStationary = 0;
	}

	cudaErrorCheck(cudaMalloc(&device_plist, sizeof(Particle)*numpart));
	cudaErrorCheck(cudaMemcpy(device_plist, plist, sizeof(Particle)*numpart, cudaMemcpyHostToDevice));

	dim3 blocks(numpart / blockSize, 1, 1);
	dim3 threadsPerBlock(blockSize, 1, 1);

	computeVelocity << < blocks, threadsPerBlock >> > (device_plist);

	return numpart;
}

#undef blockSize
int DeviceFunctions::shutdown() {
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaErrorCheck(cudaFree(device_plist));
	free(plist);
	return 0;
}
