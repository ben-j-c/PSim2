#pragma once
#include "cuda_runtime.h"
#include <array>

typedef struct Vector3 {
	float x;
	float y;
	float z;
};

__device__ __host__ Vector3* addVector3(Vector3* dst, Vector3* src);

__device__ __host__ Vector3* scaleVector3(Vector3* dst, float src);

__device__ __host__ Vector3* subVector3(Vector3* dst, Vector3* src);

__device__ __host__ Vector3* normVector3(Vector3* dst);

__device__ __host__ float magVector3(Vector3* dst);

__device__ __host__ Vector3* copyVector3(Vector3* dst, Vector3* src);

__device__ __host__ Vector3* copyVector3(Vector3* dst, std::array<float, 3>& src);
