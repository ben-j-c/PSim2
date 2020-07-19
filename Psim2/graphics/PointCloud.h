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
		cudaGraphicsResourceGetMappedPointer((void**)&vColour_device, &vColourSize, vPos.cuda);
	}

	void draw(GLuint program) {
		glEnableVertexAttribArray(glGetAttribLocation(program, "vPos"));
		glEnableVertexAttribArray(glGetAttribLocation(program, "vColour"));
		glBindBuffer(GL_ARRAY_BUFFER, *vPos);
		glVertexAttribPointer(glGetAttribLocation(program, "vPos"),
			3, //Vector3
			GL_FLOAT, //Vector of floats
			GL_FALSE, //Do not normalize
			0, //Stride
			0);//Use binded buffer
		glBindBuffer(GL_ARRAY_BUFFER, *vColour);
		glVertexAttribPointer(glGetAttribLocation(program, "vColour"),
			3, //Vector3
			GL_FLOAT, //Vector of floats
			GL_FALSE, //Do not normalize
			0, //Stride
			0);//Use binded buffer

		glDrawArrays(GL_POINTS, 0, N);

		glDisableVertexAttribArray(glGetAttribLocation(program, "vPos"));
		glDisableVertexAttribArray(glGetAttribLocation(program, "vColour"));
	};
};

