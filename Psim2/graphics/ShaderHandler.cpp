#include "ShaderHandler.hpp"
#include "Graphics.h"
#include <fstream>
#include <string>
#include <istream>
#include <vector>
#include <unordered_map>
#include <stdlib.h>
#include <GL/glew.h>

static std::string progName;
static std::unordered_map<std::string, Program> programs;


static bool hasProgram(const std::string & s) {
	return programs.count(s) > 0;
}

/*
Create GL program
*/
void ShaderHandler::createProgram(const std::string& name) {
	if (!hasProgram(name)) {
		programs[name] = Program(glCreateProgram());
		checkGL("glCreateProgram");
	}
}

void ShaderHandler::switchProgram(const std::string& name) {
	if (name.compare("null") == 0) {
		Graphics::program = 0;
		glUseProgram(0);
		checkGL("glUseProgram, can't continue without program");
		return;
	}
	if (!hasProgram(name)) {
		throw "No program with given name: " + name + "\n";
	}

	progName = name;
	Graphics::program = programs[name].id;
	glUseProgram(Graphics::program);
	checkGL("glUseProgram, can't continue without program");
}


/*
glAttachShader on the corresponding shader
*/
int ShaderHandler::attachShader(const std::string& program, const std::string& fileName) {
	Program& p = programs[program];
	if (p.hasShader(fileName)) {
		p.shaders[fileName].isActive = 1;
		glAttachShader(p.id, p.shaders[fileName].shader);
		checkGL("glAttachShader");
		return 0;
	}
	else
		fprintf(stderr, "Shader %20s has not been loaded\n", fileName.c_str());
	return 1;
}

/*
Load a shader file and store in the given shader
*/
static int readShaderFile(Shader* shader) {
	std::ifstream shaderStream(shader->fileName);

	if (shaderStream.is_open()) {
		shader->lines = (char**)calloc(64, sizeof(char*));
		int maxNumLines = 64, curLine = 0;
		while (shaderStream.good()) {
			if (curLine == maxNumLines)
				realloc(shader->lines, (maxNumLines *= 2) * sizeof(char*));
			std::string line;
			std::getline(shaderStream, line);
			line += "\n";
			shader->lines[curLine] = (char*)calloc(line.length() + 1, sizeof(char*));
			strcpy_s(
				shader->lines[curLine], line.length() + 1, line.c_str());
			curLine++;
		}
		shader->numLines = curLine;
		shaderStream.close();
	}
	else {
		std::cerr << "Cannot read shader file: " << shader->fileName << std::endl;
		return 1;
	}
	return 0;
}

/*
glShaderSource, glCompileShader

Reads the shader from the file given into a shader object and makes OpenGL aware of it.
*/
int ShaderHandler::loadFragmentShader(const std::string& program, const std::string& fileName) {
	Program& p = programs[program];
	if (!p.hasShader(fileName)) {
		Shader newShader(fileName.c_str());
		newShader.shader = glCreateShader(GL_FRAGMENT_SHADER);
		int error = readShaderFile(&newShader);
		if (!error) {
			glShaderSource(newShader.shader, newShader.numLines, newShader.lines, nullptr);
			glCompileShader(newShader.shader);
			checkGL("glCompileShader on fragment");

			char* infolog = (char*)malloc(sizeof(char) * 10000);
			int length;
			glGetShaderInfoLog(newShader.shader, 10000, &length, infolog);
			printf("<Fragment shader info>\n");
			printf("%s", infolog);
			printf("<Fragment shader end info>\n");
			free(infolog);

			p.shaders[fileName] = std::move(newShader);
			return 0;
		}
	}
	return 1;
}

/*
glShaderSource, glCompileShader

Reads the shader from the file given into a shader object and makes OpenGL aware of it.
*/
int ShaderHandler::loadVertexShader(const std::string& program, const std::string& fileName) {
	Program& p = programs[program];
	if (!p.hasShader(fileName)) {
		Shader newShader(fileName.c_str());
		newShader.shader = glCreateShader(GL_VERTEX_SHADER);
		int error = readShaderFile(&newShader);
		if (!error) {
			glShaderSource(newShader.shader, newShader.numLines, newShader.lines, nullptr);
			glCompileShader(newShader.shader);
			checkGL("glCompileShader on vertex shader");

			char* infolog = (char*)malloc(sizeof(char) * 10000);
			int length;
			glGetShaderInfoLog(newShader.shader, 10000, &length, infolog);
			printf("<Vertex shader info>\n");
			printf("%s", infolog);
			printf("<Vertex shader end info>\n");
			free(infolog);

			p.shaders[fileName] = std::move(newShader);
			return 0;
		}
	}
	return 1;
}

int ShaderHandler::linkShaders(const std::string& program) {
	glLinkProgram(programs[program].id);
	char* infolog = (char*)malloc(sizeof(char) * 10000);
	int length;
	glGetProgramInfoLog(programs[program].id, 10000, &length, infolog);
	printf("<Link shader info>\n");
	printf("%s", infolog);
	printf("<Link shader end info>\n");
	free(infolog);
	checkGL("link vertex shader");
	return 0;
}

/*
Called at the end of a program.

glDetachShader and deallocate the shader.
*/
void ShaderHandler::shutdownShaders(const std::string& program) {
	Program& p = programs[program];
	for (auto it = p.shaders.begin(); it != p.shaders.end(); ++it) {
		detachShader(program, it->second.fileName);
		deleteShader(program, it->second.fileName);
	}
	programs.erase(program);
}

/*
glDetachShader
*/
void ShaderHandler::detachShader(const std::string& program, const std::string& fileName) {
	Program& p = programs[program];
	if (p.hasShader(fileName) && p.shaders[fileName].isActive) {
		p.shaders[fileName].isActive = 0;
		glDetachShader(p.id, p.shaders[fileName].shader);
	}
}

/*

*/
void ShaderHandler::deleteShader(const std::string& program, const std::string& fileName) {
	Program& p = programs[program];
	if (p.hasShader(fileName)) {
		if (p.shaders[fileName].isActive) {
			detachShader(program, fileName);
		}
		glDeleteShader(p.shaders[fileName].shader);
	}
}

Shader::Shader(const char* file) {
	fileName = std::string(file);
}

Shader::Shader() {
}

Shader& Shader::operator=(Shader && toMove) {
	fileName = std::move(toMove.fileName);
	shader = toMove.shader;
	lines = toMove.lines;
	numLines = toMove.numLines;
	isActive = toMove.isActive;

	toMove.shader = 0;
	toMove.lines = nullptr;
	toMove.numLines = 0;
	toMove.isActive = false;
	return *this;
}

/*
*/
Shader::~Shader() {
	for (int i = 0; i < this->numLines; i++) {
		free(this->lines[i]);
	}

	if(lines)
		free(this->lines);
}

/*
Just search for an existing shader by file name.
*/
Shader& Program::findShader(const std::string& s) {
	return shaders[s];
}

bool Program::hasShader(const std::string& s) {
	return shaders.count(s) > 0;
}