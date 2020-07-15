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

	void draw();
};

