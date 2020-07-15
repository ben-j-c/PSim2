#include "vec4.hpp"
#include "math.h"

using namespace Graphics;

vec4::vec4(){
	;
}

vec4::vec4(float a, float b, float c, float d){
	value[0] = a;
	value[1] = b;
	value[2] = c;
	value[3] = d;
}

vec4::vec4(float newValue[4]){
	value[0] = newValue[0];
	value[1] = newValue[1];
	value[2] = newValue[2];
	value[3] = newValue[3];
}

vec4::vec4(float newValue){
	value[0] = newValue;
	value[1] = newValue;
	value[2] = newValue;
	value[3] = newValue;
}

vec4& vec4::operator=(const vec4& other){
	for(int i = 0 ; i < 4 ; i++)
		value[i] = other.value[i];
	return *this;
}

vec4 vec4::operator+(const vec4& other){
	vec4 returner;
	for(int i = 0; i < 4 ; i++)
		returner.value[i] = value[i] + other.value[i];
	return returner;
}

vec4 vec4::operator-(const vec4& other){
	vec4 returner;
	for(int i = 0; i < 4 ; i++)
		returner.value[i] = value[i] - other.value[i];
	return returner;
}

float vec4::operator*(const vec4& other){
	return 
	   value[0]*other.value[0]
	 + value[1]*other.value[1]
	 + value[2]*other.value[2]
	 + value[3]*other.value[3];
}

float vec4::dot(const vec4& other){
	return operator*(other);
}

vec4 vec4::cross(const vec4& other){
	vec4 returner;
	returner.value[0] = value[1]*other.value[2] - value[2]*other.value[1];
	returner.value[1] = -value[0]*other.value[2] + value[2]*other.value[0];
	returner.value[2] = value[0]*other.value[1] - value[1]*other.value[0];
	return returner;
}

vec4& vec4::operator+=(const vec4& other){
	for(int i = 0; i<4 ;i++)
		value[i] += other.value[i];
	return *this;
}

vec4& vec4::operator-=(const vec4& other){
	for(int i = 0; i<4 ;i++)
		value[i] -= other.value[i];
	return *this;
}

vec4& vec4::operator*=(const vec4& other){
	for(int i = 0; i<4 ;i++)
		value[i] *= other.value[i];
	return *this;
}

vec4 vec4::operator+(float other){
	vec4 returner(other);
	returner += *this;
	return returner;
}

vec4 vec4::operator-(float other){
	vec4 returner(-other);
	returner += *this;
	return returner;
}

vec4 vec4::operator*(float other){
	vec4 returner(other);
	returner *= *this;
	return returner;
}

vec4 vec4::operator/(float other){
	vec4 returner(*this);
	returner /= other;
	return returner;
}

vec4& vec4::operator+=(float other){
	for(int i = 0; i<4 ;i++)
		value[i] += other;
	return *this;
}

vec4& vec4::operator-=(float other){
	for(int i = 0; i<4 ;i++)
		value[i] -= other;
	return *this;
}

vec4& vec4::operator*=(float other){
	for(int i = 0; i<4 ;i++)
		value[i] *= other;
	return *this;
}

vec4& vec4::operator/=(float other){
	for(int i = 0; i<4 ;i++)
		value[i] /= other;
	return *this;
}

vec4 vec4::operator||(const vec4& other){
	vec4& b = (vec4&) other;
	vec4& a = (vec4&) *this;
	vec4 returner = b.unit()*(a*b.unit());
	return returner;
}

inline vec4 vec4::projectOnto(const vec4& other){
	return (*this) || other;
}

vec4 vec4::operator^(const vec4& other){
	vec4& b = (vec4&) other;
	vec4& a = (vec4&) *this;
	vec4 returner = a - b.unit()*(a*b.unit());
	return returner;
}

inline vec4 vec4::projectOrthogonal(const vec4& other){
	return (*this)^other;
}

inline vec4& vec4::unitize(){
	return operator/=(magnitude());
}

vec4 vec4::unit(){
	vec4 returner(*this);
	returner /= magnitude();
	return returner;
}

float vec4::magnitude(){
	return sqrt(dot(*this));
}

float vec4::mag(){
	return magnitude();
}

float vec4::magnitudeSquared(){
	return dot(*this);
}

float vec4::magSq(){
	return magnitudeSquared();
}

vec4& vec4::rotateX(float theta) {
	vec4 newPos(
		value[0],
		value[1]*cosf(theta) - value[2]*sinf(theta),
		value[2]*cosf(theta) + value[1]*sinf(theta),
		value[3]);
	*this = newPos;
	return *this;
}
vec4& vec4::rotateY(float theta) {
	vec4 newPos(
		value[0]*cosf(theta) + value[2]*sinf(theta),
		value[1],
		value[2]*cosf(theta) - value[0]*sinf(theta),
		value[3]);
	*this = newPos;
	return *this;
}
vec4& vec4::rotateZ(float theta) {
	vec4 newPos(
		value[0]*cosf(theta) - value[1]*sinf(theta),
		value[1]*cosf(theta) + value[0]*sinf(theta),
		value[2],
		value[3]);
	*this = newPos;
	return *this;
}

vec4location vec4::operator[](int row){
	vec4location returner(this, row);
	return returner;
}

vec4location vec4::operator()(int row){
	vec4location returner(this, row);
	return returner;
}

vec4::operator std::string(){
	char outputStream[256];
	sprintf_s(outputStream, 256, "{ %f %f %f %f }", value[0], value[1], value[2], value[3]);
	return std::string(outputStream);;
}