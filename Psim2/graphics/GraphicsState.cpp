#define MODEL_MATRIX
#define _USE_MATH_DEFINES

#include <GL/glew.h>
#include <math.h>
#include <iostream>
#include <unordered_map>
#include <stack>
#include "GraphicsState.hpp"
#include "TextureHandler.h"
#include "math/mat4.hpp"
#include "math/vec4.hpp"


class FrameBuffer {
public:
	Graphics::GLResource FBO;

	FrameBuffer() : FBO(new GLuint, Graphics::FrameBufferDeleter) {
		glGenFramebuffers(1, FBO.get());
		checkGL("glGenFramebuffers");
	}
};

static std::unordered_map<std::string, FrameBuffer> frameBuffers;
static std::stack<Graphics::mat4> modelStack;
static Graphics::mat4 model(true);

namespace Graphics {
	static inline void updateModel() {
		int32_t m = glGetUniformLocation(Graphics::program, "modelMatrix");
		if (m != -1) {
			GLfloat* params = model.data();
			glUniformMatrix4fv(m, 1, true, params);
		}
		else {
			throw std::runtime_error("could not find modelMatrix in shader");
		}
	}

	static inline void updateProjection(const Graphics::mat4& newMatrix) {
		int32_t  projection = glGetUniformLocation(Graphics::program, "projectionMatrix");
		if (projection != -1) {
			Graphics::mat4& m = (Graphics::mat4&) newMatrix;
			GLfloat *params = m.getCopy();
			glUniformMatrix4fv(projection, 1, true, params);
			free(params);
		}
		else {
			throw std::runtime_error("projectionMatrix not found in shader");
		}
	}

	static inline void updateView(const Graphics::mat4& newMatrix) {
		int32_t  view = glGetUniformLocation(Graphics::program, "viewMatrix");
		if (view != -1) {
			Graphics::mat4& m = (Graphics::mat4&) newMatrix;
			GLfloat *params = m.getCopy();
			glUniformMatrix4fv(view, 1, true, params);
			free(params);
			checkGL("updateView");
		}
		else {
			throw std::runtime_error("viewMatrix not found in shader");
		}
	}


	/*
		Link the uniform locations within the shaders.
	*/

#ifdef LIGHTING
	void State::setDiffuse(float value) {
		glUniform1f(Kd, value);
	}

	void State::setSpecular(float value) {
		glUniform1f(Ks, value);
	}

	void State::setAmbient(float value) {
		glUniform1f(Ka, value);
	}

	void State::setAmbientColour(float r, float g, float b) {
		glUniform3f(ambientColour, r, g, b);
	}

	void State::setLightPosition(float x, float y, float z) {
		glUniform4f(lightPos, x, y, z, 1);
	}

	void State::setLightColour(float r, float g, float b) {
		glUniform3f(lightColour, r, g, b);
	}

	void State::setShine(float value) {
		glUniform1f(shine, value);
	}
#endif


	void loadIdentity() {
		model = mat4(true);
		updateModel();
	}

	void translate(float x, float y, float z) {
		float translate[16] = {
			1.0,0.0,0.0,x,
			0.0,1.0,0.0,y,
			0.0,0.0,1.0,z,
			0.0,0.0,0.0,1.0
		};
		model *= translate;
		updateModel();
	}

	void rotate(float theta, float x, float y, float z) {
		float n[16] = {
			0.0, -z,  y,   0.0,
			z,   0.0, -x,  0.0,
			-y,  x,   0.0, 0.0,
			0.0, 0.0, 0.0, 1.0
		};

		float rad = theta * (float)M_PI / 180.0f;
		mat4 ncross(n);
		mat4 b = ncross*sinf(rad);
		mat4 c = ncross * ncross*(1 - cosf(rad));
		b(3, 3) = 0;
		c(3, 3) = 0;
		model *= b + c + mat4::Identity;
		updateModel();
	}

	void rotatex(float theta) {
		rotate(theta, 1, 0, 0);
		updateModel();
	}

	void rotatey(float theta) {
		rotate(theta, 0, 1, 0);
		updateModel();
	}

	void rotatez(float theta) {
		float rad = theta * (float)M_PI / 180.0f;
		float c = cosf(rad);
		float s = sinf(rad);
		float rot[16] = {
			  c,    -s, 0.0f, 0.0f,
			  s,     c, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f,
		};
		model *= rot;
		updateModel();
	}

	void scale(float x, float y, float z) {
		float scale[16] = {
			x,   0.0, 0.0, 0.0,
			0.0, y,   0.0, 0.0,
			0.0, 0.0, z,   0.0,
			0.0, 0.0, 0.0, 1.0
		};
		model *= scale;
		updateModel();
	}

	void pose(float x, float y, float theta, float sx, float sy) {
		float rad = theta * (float)M_PI / 180.0f;
		float pose[16] = {
		sx*cosf(rad), -sy*sinf(rad), 0, x,
		sx*sinf(rad),  sy*cosf(rad), 0, y,
		0, 0, 1.0f, 0,
		0, 0, 0, 1.0f };
		model *= pose;
		updateModel();
	}

	void setModelPose2D(float x, float y, float theta, float sx, float sy) {
		float rad = theta * (float)M_PI / 180.0f;
		float pose[16] = {
		sx*cosf(rad), -sy * sinf(rad), 0, x,
		sx*sinf(rad),  sy*cosf(rad), 0, y,
		0, 0, 1.0f, 0,
		0, 0, 0, 1.0f };
		int32_t m = glGetUniformLocation(Graphics::program, "modelMatrix");
		if (m != -1) {
			glUniformMatrix4fv(m, 1, true, pose);
		}
		else {
			throw std::runtime_error("could not find modelMatrix in shader");
		}
	}

	void setCamera(float fov, float ratio, float near, float far, const vec4& cameraPos, const vec4& cameraLookAt, const vec4& cameraUpVector) {
		float f = (1 / tanf(fov / 180.0f* (float)M_PI / 2));
		float d = far - near;

		float pMatrix[16] = {
			f / ratio,   0.0,       0.0,            0.0,
			0.0,       f,         0.0,            0.0,
			0.0,       0.0,      -(near + far) / d,  -2 * near*far / d,
			0.0,       0.0,      -1.0,            0.0
		};

		mat4 projectionMatrix(pMatrix);

		vec4& pos = (vec4&)cameraPos;
		vec4& lookAt = (vec4&)cameraLookAt;
		vec4& up = (vec4&)cameraUpVector;

		//Construct basis vectors
		vec4 k = (pos - lookAt) / (pos - lookAt).magnitude();
		vec4 i = up.cross(k) / up.cross(k).magnitude();
		vec4 j = k.cross(i);

		//Inverse camera matrix; transforms world coordinate system into viewing coordinate system

		float A[16] = {
			i(0),i(1),i(2),0.0,
			j(0),j(1),j(2),0.0,
			k(0),k(1),k(2),0.0,
			0.0, 0.0, 0.0, 1.0
		};
		float B[] = {
			1.0, 0.0, 0.0, -pos(0),
			0.0, 1.0, 0.0, -pos(1),
			0.0, 0.0, 1.0, -pos(2),
			0.0, 0.0, 0.0, 1.0
		};

		mat4 Mcam_inv = mat4(A)*mat4(B);

		updateProjection(projectionMatrix);
		updateView(Mcam_inv);
	}

	void setCamera2D(float ratio, float span, vec4 pos) {
		float pMatrix[16] = {
			2.0f / (span*ratio), 0.0f,  0.0f, 0.0f,
			0.0f, 2.0f / span, 0.0f, 0.0f,
			0.0f, 0.0f,  1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		};
		float B[] = {
			1.0f, 0.0f, 0.0f, -pos(0),
			0.0f, 1.0f, 0.0f, -pos(1),
			0.0f, 0.0f, 1.0f, -pos(2),
			0.0f, 0.0f, 0.0f, 1.0f
		};

		mat4 viewMatrix(pMatrix);
		viewMatrix *= mat4(B);
		updateView(viewMatrix);
	}

	void pushModel() {
		modelStack.push(model);
	}

	void popModel() {
		model = modelStack.top();
		modelStack.pop();
		updateModel();
	}

	void createFrameBuffer(const std::string & frameName,
		const std::string & tex,
		int attachment,
		int width,
		int height) {
		if (frameBuffers.count(frameName)) {
			throw std::runtime_error("Named frame buffer already exists");
		}
		if (!TextureHandler::makeTexture(tex, width, height)) {
			throw std::runtime_error("Named texture already exists \"" + frameName + "\"");
		}
		FrameBuffer fb;
		frameBuffers[frameName] = fb;
		glBindFramebuffer(GL_FRAMEBUFFER, *fb.FBO);
		glFramebufferTexture(GL_FRAMEBUFFER,
			GL_COLOR_ATTACHMENT0 + attachment,
			TextureHandler::getTextureHandle(tex),
			0);
	}

	void useFrameBuffer(const std::string & frameName) {
		if (!frameBuffers.count(frameName)) {
			throw std::runtime_error("Named frame buffer does not exist");
		}
		glBindFramebuffer(GL_FRAMEBUFFER, *frameBuffers[frameName].FBO);
	}

	void useDefaultFrameBuffer() {
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void setUniform(const std::string & name, const int & val) {
		GLint loc = glGetUniformLocation(Graphics::program, name.c_str());
		if (loc != -1) {
			glUniform1i(loc, val);
		}
		else {
			throw std::runtime_error("Could not find uniform \"" + name + "\"");
		}
	}
}