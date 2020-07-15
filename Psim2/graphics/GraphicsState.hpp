#pragma once
#include "math\mat4.hpp"
#include "Graphics.h"

namespace Graphics {
	void loadIdentity();
	void translate(float x, float y, float z);
	void rotate(float theta, float x, float y, float z);
	void rotatex(float theta);
	void rotatey(float theta);
	void rotatez(float theta);
	void scale(float x, float y, float z);
	void pose(float x, float y, float theta, float sx, float sy);
	void setModelPose2D(float x, float y, float theta, float sx, float sy);
	void applyMatrix(const mat4&);
	void setCamera(float fov, float ratio, float near, float far, const vec4& pos, const vec4& lookAt, const vec4& up);
	void setCamera2D(float ratio, float span, vec4 pos);
	void setDiffuse(float value);
	void setSpecular(float value);
	void setAmbient(float value);
	void setAmbientColour(float r, float g, float b);
	void setLightPosition(float x, float y, float z);
	void setLightColour(float r, float g, float b);
	void setShine(float value);

	void pushModel();
	void popModel();

	void createFrameBuffer(const std::string& frameName,
		const std::string& tex,
		int attachment,
		int width,
		int height);
	void useFrameBuffer(const std::string& frameName);
	void useDefaultFrameBuffer();
	void setUniform(const std::string& name, const int& val);
}