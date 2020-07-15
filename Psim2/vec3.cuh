#ifndef HOST_DEVICE_VEC3
#define HOST_DEVICE_VEC3
#include "cuda_runtime.h"

#define CUDA_GEN_BOTH __device__ __host__

class vec3 {
public:
	float x, y, z;

	CUDA_GEN_BOTH vec3 operator+(const vec3&);
	CUDA_GEN_BOTH vec3 operator-(const vec3&);
	CUDA_GEN_BOTH vec3 operator*(const vec3&);
	CUDA_GEN_BOTH vec3 operator+(float);
	CUDA_GEN_BOTH vec3 operator-(float);
	CUDA_GEN_BOTH vec3 operator*(float);
	CUDA_GEN_BOTH vec3 operator/(float);

	CUDA_GEN_BOTH vec3& operator+=(const vec3&);
	CUDA_GEN_BOTH vec3& operator-=(const vec3&);
	CUDA_GEN_BOTH vec3& operator*=(const vec3&);
	CUDA_GEN_BOTH vec3& operator+=(float);
	CUDA_GEN_BOTH vec3& operator-=(float);
	CUDA_GEN_BOTH vec3& operator*=(float);
	CUDA_GEN_BOTH vec3& operator/=(float);

	CUDA_GEN_BOTH float dot(const vec3&);
	CUDA_GEN_BOTH vec3 cross(const vec3&);
	CUDA_GEN_BOTH float mag();
	CUDA_GEN_BOTH float magSq();
	CUDA_GEN_BOTH float magnitude() { return mag(); };
	CUDA_GEN_BOTH float magnitudeSquared() { return magSq(); };
	CUDA_GEN_BOTH vec3& flipSign();
	CUDA_GEN_BOTH bool isZero(float epsilon = 0.0f);
	CUDA_GEN_BOTH bool isEqual(const vec3&, float epsilon = 0.0f);
	CUDA_GEN_BOTH vec3& clampMag(float mag);
};


CUDA_GEN_BOTH vec3 operator+(float, const vec3&);
CUDA_GEN_BOTH vec3 operator-(float, const vec3&);
CUDA_GEN_BOTH vec3 operator*(float, const vec3&);

#endif