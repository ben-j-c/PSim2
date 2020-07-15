#include "mat4.hpp"
#include <math.h>
#include <iostream>
#include <stdio.h>

using namespace Graphics;

//static int globalIndex = 0;

const mat4 mat4::Identity(true);

mat4::mat4(bool identity){
	if (identity) {
		for (int i = 0; i < 4; i++) {
			value[i][i] = 1.0f;
		}
	}
	//std::cout << "allocated " << (this->index = globalIndex++) << std::endl;
}

mat4::mat4(float newValue) : mat4(){
	for(int i = 0;i < 4;i++){
		for(int j = 0;j < 4;j++){
			value[i][j] = newValue;
		}
	}
}

mat4::mat4(const mat4& newValue) : mat4(){
	for(int i = 0;i < 4;i++){
		for(int j = 0;j < 4;j++){
			value[i][j] = newValue.value[i][j];
		}
	}
}

mat4::mat4(const vec4& a, const vec4& b, const vec4& c, const vec4& d) : mat4(){
	for(int i = 0;i < 4;i++){
		value[i][0] = ((vec4&) a).at(i);
	}
	for(int i = 0;i < 4;i++){
		value[i][1] = ((vec4&) b).at(i);
	}
	for(int i = 0;i < 4;i++){
		value[i][2] = ((vec4&) c).at(i);
	}
	for(int i = 0;i < 4;i++){
		value[i][3] = ((vec4&) d).at(i);
	}
}

mat4::mat4(float newValue[4][4]) : mat4(){
	for(int i = 0;i < 4;i++){
		for(int j = 0; j < 4;j++){
			value[i][j] = newValue[i][j];
		}
	}
}

mat4::mat4(float newValue[16]) : mat4(){
	for(int i = 0;i < 16;i++){
		value[i/4][i%4] = newValue[i];
	}
}

mat4::~mat4(){
	//std::cout << "deallocated " << this->index << std::endl;
}

mat4 mat4::operator+(const mat4& add){
	mat4 returner(add);
	for(int i = 0;i < 4;i++){
		for(int j = 0;j < 4;j++){
			returner(i, j) += this->at(i, j);
		}
	}
	return returner;
}

mat4 mat4::operator-(const mat4& add) {
	mat4 returner(add);
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			returner(i, j) -= this->at(i, j);
		}
	}
	return returner;
}

mat4& mat4::operator+=(const mat4& add){
	for(int i = 0;i < 4;i++){
		for(int j = 0;j < 4;j++){
			value[i][j] += ((mat4&) add).at(i,j);
		}
	}
	return *this;
}

mat4& mat4::operator-=(const mat4& add) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			value[i][j] -= ((mat4&)add).at(i, j);
		}
	}
	return *this;
}

mat4 mat4::operator*(const mat4& mult){
	mat4& a = *this;
	mat4& b = (mat4&) mult;
	mat4 returner;
	for(int i = 0;i < 4;i++){
		for(int j = 0;j < 4;j++){
			returner(i, j) =  a(i,0)*b(0,j)
							+ a(i,1)*b(1,j)
							+ a(i,2)*b(2,j)
							+ a(i,3)*b(3,j);
		}
	}
	return returner;
}

mat4 Graphics::mat4::operator*(const float values[16]) {
	mat4& a = *this;
	mat4 returner;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			returner(i, j) = a(i, 0)*values[j]
				+ a(i, 1)*values[j + 4]
				+ a(i, 2)*values[j + 8]
				+ a(i, 3)*values[j + 12];
		}
	}
	return returner;
}

mat4 Graphics::mat4::operator*(const float values[4][4]) {
	mat4& a = *this;
	mat4 returner;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			returner(i, j) = a(i, 0)*values[0][j]
				+ a(i, 1)*values[1][j]
				+ a(i, 2)*values[2][j]
				+ a(i, 3)*values[3][j];
		}
	}
	return returner;
}

mat4& mat4::operator*=(const mat4& b) {
	this->operator=((*this)*b);
	return *this;
}

mat4& mat4::operator=(const mat4& other){
	for(int i = 0 ; i < 4 ; i++){
		for(int j = 0 ; j < 4 ; j++){
			value[i][j] = other.value[i][j];
		}
	}
	return *this;
}

mat4 mat4::operator+(float value){
	mat4 returner(value);
	returner += *this;
	return returner;
}

mat4& mat4::operator+=(float value){
	for(int i = 0 ; i < 4 ; i++){
		for(int j = 0 ; j < 4 ; j++){
			this->value[i][j] += value;
		}
	}
	return *this;
}

mat4 mat4::operator*(float value){
	mat4 returner(*this);
	for(int i = 0 ; i < 4 ; i++){
		for(int j = 0 ; j < 4 ; j++){
			returner.value[i][j] *= value;
		}
	}
	return returner;
}

vec4 mat4::operator*(const vec4 &other){
	vec4 returner;
	vec4& oth = (vec4&) other;
	for(int i = 0 ; i < 4 ; i++){
		returner.value[i] = 	  oth.at(0)*value[i][0]
								+ oth.at(1)*value[i][1]
								+ oth.at(2)*value[i][2]
								+ oth.at(3)*value[i][3];
	}
	return returner;
}

mat4& mat4::operator*=(float value){
	for(int i = 0 ; i < 4 ; i++){
		for(int j = 0 ; j < 4 ; j++){
			this->value[i][j] *= value;
		}
	}
	return *this;
}

mat4& mat4::operator=(float value){
	for(int i = 0 ; i < 4 ; i++){
		for(int j = 0 ; j < 4 ; j++){
			this->value[i][j] = value;
		}
	}
	return *this;
}

mat4location mat4::operator[](int index) {
	mat4location returner(this, index%4, index/4);
	return returner;
}

mat4location mat4::operator()(int row, int col) {
	mat4location returner(this, row, col);
	return returner;
}

inline const float mat4::at(int row, int col) {
	return value[row][col];
}

inline const void mat4::set(int row, int col, float setValue) {
	value[row][col] = setValue;
}

float *mat4::getCopy(){
	float* returner = (float*) malloc(sizeof(float)*16);
	for(int i = 0;i<4;i++){
		for(int j = 0;j < 4;j++){
			*(returner + i*4 + j) = value[i][j];
		}
	}
	return returner;
}

float* Graphics::mat4::data() {
	return (float*)value;
}

mat4 Graphics::getIdentity(){
	mat4 returner;
	returner(0,0) = 1;
	returner(1,1) = 1;
	returner(2,2) = 1;
	returner(3,3) = 1;
	return returner;
}

void Graphics::printMat4(const mat4& toPrint){
	mat4& p = (mat4&) toPrint;
	for(int i = 0;i<4;i++)
		printf("%f %f %f %f\n", p.at(i,0), p.at(i,1), p.at(i,2), p.at(i,3));
}