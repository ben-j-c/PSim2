#include "Graphics.h"
#include <sstream>

GLuint Graphics::program;

std::function<void(GLuint*)> Graphics::BuffDeleter = [](GLuint* i) {
	glDeleteBuffers(1, i);
	delete i;
};

std::function<void(GLuint*)> Graphics::FrameBufferDeleter = [](GLuint* i) {
	glDeleteFramebuffers(1, i);
	delete i;
};

std::function<void(GLuint*)> Graphics::ProgramDeleter = [](GLuint* i) {
	glDeleteProgram(*i);
	delete i;
};

std::function<void(GLuint*)> Graphics::RenderBufferDeleter = [](GLuint* i) {
	glDeleteRenderbuffers(1, i);
	delete i;
};

std::function<void(GLuint*)> Graphics::ShaderDeleter = [](GLuint* i) {
	glDeleteShader(*i);
	delete i;
};

std::function<void(GLuint*)> Graphics::TexDeleter = [](GLuint* i) {
	glDeleteTextures(1, i);
	delete i;
};

std::function<void(GLuint*)> Graphics::VAODeleter = [](GLuint* i) {
	glDeleteVertexArrays(1, i);
	delete i;
};

void Graphics::checkOGLError(const char* info, const char* file, int line, const char* function) {
	GLenum error = glGetError();
	if (error != GL_NO_ERROR) {
		std::stringstream ss;
		ss << "OpenGL error " << std::hex << error << std::dec << ". (" << info <<") in file " << file << " " << line << ":" << function << std::endl;
		throw std::runtime_error(std::string(ss.str()));
	}
}

void GLAPIENTRY Graphics::MessageCallback(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar* message,
	const void* userParam) {
	if (severity == GL_DEBUG_SEVERITY_NOTIFICATION)
		return;

	fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
		(type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
		type, severity, message);
}