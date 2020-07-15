#ifndef SHADER_HANDLER
#define SHADER_HANDLER
#include <string.h>
#include <iostream>
#include <unordered_map>
#include "Graphics.h"

namespace ShaderHandler {
	void createProgram(const std::string& name);
	void switchProgram(const std::string& name);
	int attachShader(const std::string& program, const std::string& name);
	int loadVertexShader(const std::string& program, const std::string& name);
	int loadFragmentShader(const std::string& program, const std::string& name);
	int linkShaders(const std::string& program);
	void deleteShader(const std::string& program, const std::string& name);
	void detachShader(const std::string& program, const std::string& name);
	void shutdownShaders(const std::string& program);
}

class Shader {
public:
	std::string fileName;
	unsigned int shader;
	char** lines;
	int numLines = 0;
	int isActive = 0;
	Shader(const char*);
	Shader();
	Shader & operator=(Shader && toMove);
	~Shader();
};

class Program {
public:
	Program(uint32_t id) : id(id) {};
	Program() = default;
	Program(Program&&) = default;
	Program& operator=(Program&&) = default;

	Program(const Program&) = delete;
	Program& operator=(const Program&) = delete;

	std::unordered_map<std::string, Shader> shaders;
	uint32_t id;

	Shader& findShader(const std::string& fileName);
	bool hasShader(const std::string& fileName);
};

#endif