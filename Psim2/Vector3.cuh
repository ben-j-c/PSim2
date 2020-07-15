#ifndef DEVICE_VECTOR_OPS
#define DEVICE_VECTOR_OPS
#include "cuda_runtime.h"

typedef struct Vector3 {
	float x;
	float y;
	float z;
} Vector3;

__device__ Vector3* addVector3(Vector3* dst, Vector3* src);

__device__ Vector3* scaleVector3(Vector3* dst, float src);

__device__ Vector3* subVector3(Vector3* dst, Vector3* src);

__device__ Vector3* normVector3(Vector3* dst);

__device__ float magVector3(Vector3* dst);

__device__ Vector3* copyVector3(Vector3* dst, Vector3* src);

#endif // !DEVICE_VECTOR_OPS
