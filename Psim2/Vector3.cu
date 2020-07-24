#include "Vector3.cuh"
#include <cuda_runtime.h>

#ifdef __INTELLISENSE__
//#define __CUDACC__
#include <math_functions.h>
#endif // __INTELLISENSE__



Vector3* addVector3(Vector3* dst, Vector3* src) {
	dst->x = dst->x + src->x;
	dst->y = dst->y + src->y;
	dst->z = dst->z + src->z;
	return dst;
}

Vector3* scaleVector3(Vector3* dst, float src) {
	dst->x = dst->x * src;
	dst->y = dst->y * src;
	dst->z = dst->z * src;
	return dst;
}

Vector3* subVector3(Vector3* dst, Vector3* src) {
	dst->x = dst->x + src->x;
	dst->y = dst->y + src->y;
	dst->z = dst->z + src->z;
	return dst;
}

Vector3* normVector3(Vector3* dst) {
	float invSqrt = rsqrtf(dst->x*dst->x + dst->y*dst->y + dst->z*dst->z);
	return scaleVector3(dst, invSqrt);
}

float magVector3(Vector3* dst) {
	return sqrtf(dst->x*dst->x + dst->y*dst->y + dst->z*dst->z);
}

Vector3* copyVector3(Vector3* dst, Vector3* src) {
	dst->x = src->x;
	dst->y = src->y;
	dst->z = src->z;
	return dst;
}

Vector3* copyVector3(Vector3* dst, std::array<float, 3>& src) {
	dst->x = src[0];
	dst->y = src[1];
	dst->z = src[2];
	return dst;
}