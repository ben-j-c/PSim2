#pragma once
#include "graphics/PointCloud.h"

class Model {
public:
	PointCloud cloud;
	GLuint program;

	Model(size_t N, GLuint program);
	~Model();
	void step();
	void draw();
	static void releaseCuda();

};