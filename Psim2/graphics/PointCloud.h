#pragma once
#include "Graphics.h"
#include <cuda_gl_interop.h>
#include "../vec3.cuh"


/* All neccessary handles for OpenGL VBOs and cuda resources. Uses RAII to handle initialization.
*/
class PointCloud {
public:
	Graphics::VBObject_CUDA vPos;
	Graphics::VBObject_CUDA vColour;
	size_t N;

	vec3 *vPos_device;
	vec3 *vColour_device;
	size_t vPosSize;
	size_t vColourSize;
	

	PointCloud(size_t N) :
		N(N),
		vPos(N*sizeof(vec3), cudaGraphicsMapFlagsWriteDiscard),
		vColour(N*sizeof(vec3), cudaGraphicsMapFlagsWriteDiscard) { 

		cudaGraphicsResourceGetMappedPointer((void**)&vPos_device, &vPosSize, vPos.cuda);
		cudaGraphicsResourceGetMappedPointer((void**)&vColour_device, &vColourSize, vColour.cuda);
	}

	void debugGraphics(GLuint program) {
		auto pos = glGetAttribLocation(program, "vPos");
		auto colour = glGetAttribLocation(program, "vColour");
		
		float vPos[9] = {
			0,0,0,
			1,0,0,
			0,1,0
		};

		float vColour[9] = {
			1,0,0,
			1,0,0,
			1,0,0
		};

		Graphics::VBObject vPosTemp;
		Graphics::VBObject vColourTemp;

		glEnableVertexAttribArray(pos);
		glEnableVertexAttribArray(colour);
		glBindBuffer(GL_ARRAY_BUFFER, *vPosTemp);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*9, vPos, GL_STATIC_DRAW);
		glVertexAttribPointer(pos,
			3, //Vector3
			GL_FLOAT, //Vector of floats
			GL_FALSE, //Do not normalize
			0, //Stride
			0);
		glBindBuffer(GL_ARRAY_BUFFER, *vColour);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 9, vColour, GL_STATIC_DRAW);
		glVertexAttribPointer(colour,
			3, //Vector3
			GL_FLOAT, //Vector of floats
			GL_FALSE, //Do not normalize
			0, //Stride
			0);

		glDrawArrays(GL_TRIANGLES, 0, 3);

		glDisableVertexAttribArray(pos);
		glDisableVertexAttribArray(colour);
	}

	void draw(GLuint program) {
		if (_DEBUG)
			debugGraphics(program);

		auto pos = glGetAttribLocation(program, "vPos");
		auto colour = glGetAttribLocation(program, "vColour");

		glEnableVertexAttribArray(pos);
		glEnableVertexAttribArray(colour);
		glBindBuffer(GL_ARRAY_BUFFER, *vPos);
		glVertexAttribPointer(pos,
			3, //Vector3
			GL_FLOAT, //Vector of floats
			GL_FALSE, //Do not normalize
			0, //Stride
			0);//Use binded buffer
		glBindBuffer(GL_ARRAY_BUFFER, *vColour);
		glVertexAttribPointer(colour,
			3, //Vector3
			GL_FLOAT, //Vector of floats
			GL_FALSE, //Do not normalize
			0, //Stride
			0);//Use binded buffer

		glDrawArrays(GL_POINTS, 0, N);

		glDisableVertexAttribArray(pos);
		glDisableVertexAttribArray(colour);
	};
};

