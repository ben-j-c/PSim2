#include "vec3.cuh"
#include <math.h>


CUDA_GEN_BOTH vec3 vec3::operator+(const vec3& b) {
	vec3 ret(*this);
	ret += b;
	return ret;
}

CUDA_GEN_BOTH vec3 vec3::operator-(const vec3& b) {
	vec3 ret(*this);
	ret -= b;
	return ret;
}

CUDA_GEN_BOTH vec3 vec3::operator*(const vec3& b) {
	vec3 ret(*this);
	ret *= b;
	return ret;
}

CUDA_GEN_BOTH vec3 vec3::operator+(float b) {
	vec3 ret(*this);
	ret += b;
	return ret;
}

CUDA_GEN_BOTH vec3 vec3::operator-(float b) {
	vec3 ret(*this);
	ret -= b;
	return ret;
}

CUDA_GEN_BOTH vec3 vec3::operator*(float b) {
	vec3 ret(*this);
	ret *= b;
	return ret;
}

CUDA_GEN_BOTH vec3 vec3::operator/(float b) {
	vec3 ret(*this);
	ret /= b;
	return ret;
}

CUDA_GEN_BOTH vec3& vec3::operator+=(const vec3& b) {
	this->x += b.x;
	this->y += b.y;
	this->z += b.z;
	return *this;
}

CUDA_GEN_BOTH vec3& vec3::operator-=(const vec3& b) {
	this->x -= b.x;
	this->y -= b.y;
	this->z -= b.z;
	return *this;
}

CUDA_GEN_BOTH vec3& vec3::operator*=(const vec3& b) {
	this->x *= b.x;
	this->y *= b.y;
	this->z *= b.z;
	return *this;
}

CUDA_GEN_BOTH vec3& vec3::operator+=(float b) {
	this->x += b;
	this->y += b;
	this->z += b;
	return *this;
}

CUDA_GEN_BOTH vec3& vec3::operator-=(float b) {
	this->x -= b;
	this->y -= b;
	this->z -= b;
	return *this;
}

CUDA_GEN_BOTH vec3& vec3::operator*=(float b) {
	this->x *= b;
	this->y *= b;
	this->z *= b;
	return *this;
}

CUDA_GEN_BOTH vec3& vec3::operator/=(float b) {
	this->x /= b;
	this->y /= b;
	this->z /= b;
	return *this;
}

CUDA_GEN_BOTH float vec3::dot(const vec3& b) {
	return x * b.x + y * b.y + z * b.z;
}

CUDA_GEN_BOTH vec3 vec3::cross(const vec3& b) {
	return { y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x };
}

CUDA_GEN_BOTH float vec3::mag() {
	return sqrtf(x*x + y*y + z*z);
}

CUDA_GEN_BOTH float vec3::magSq() {
	return x*x + y*y + z*z;
}

CUDA_GEN_BOTH vec3& vec3::flipSign() {
	x = -x;
	y = -y;
	z = -z;
	return *this;
}

CUDA_GEN_BOTH bool vec3::isZero(float epsilon) {
	return magSq() <= epsilon*epsilon;
}

CUDA_GEN_BOTH bool vec3::isEqual(const vec3& b, float epsilon) {
	return (*this - b).magSq() <= epsilon*epsilon;
}

CUDA_GEN_BOTH vec3& vec3::clampMag(float mag) {
	if (this->magSq() > mag*mag)
		this->operator*(mag / this->mag());
	return *this;
}

CUDA_GEN_BOTH vec3 operator+(float a, const vec3& b) {
	return {a + b.x, a + b.y, a+ b.z};
}

CUDA_GEN_BOTH vec3 operator-(float a, const vec3& b) {
	return { a - b.x, a - b.y, a - b.z };
}

CUDA_GEN_BOTH vec3 operator*(float a, const vec3& b) {
	return { a*b.x, a*b.y, a*b.z };
}