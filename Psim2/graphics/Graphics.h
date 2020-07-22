#pragma once
#include <GL/glew.h>
#include <functional>
#include <memory>
#include <cuda_gl_interop.h>
#define checkGL(ans) { Graphics::checkOGLError(ans,__FILE__, __LINE__, __FUNCTION__); } 
#define cudaErrorCheck(...) { cudaErrorCheckingFunction((__VA_ARGS__), __FILE__, __LINE__); } 

static inline void cudaErrorCheckingFunction(cudaError_t error, const char* file, int line, bool abort = true) {
	if (error != cudaSuccess) {
		fprintf(stderr, "Cuda error: %s %s %d\n", cudaGetErrorString(error), file, line);
		//throw std::runtime_error("CUDA error");
	}
}

namespace Graphics {
	extern GLuint program;
	using GLResource = std::shared_ptr<GLuint>;
	using GLDeleter = std::function<void(GLuint*)>;
	extern GLDeleter BuffDeleter;
	extern GLDeleter FrameBufferDeleter;
	extern GLDeleter ProgramDeleter;
	extern GLDeleter RenderBufferDeleter;
	extern GLDeleter ShaderDeleter;
	extern GLDeleter TexDeleter;
	extern GLDeleter VAODeleter;
	extern std::function<void(cudaGraphicsResource*)> cudaVBODeleter;

	void checkOGLError(const char* info, const char* file, int line, const char* function);
	void GLAPIENTRY MessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar * message, const void * userParam);

	class OpenGLObject {
	public:
		GLResource resource;
		OpenGLObject(GLDeleter d) : resource(new GLuint, d) {};
		GLuint operator*() {
			return *resource;
		}
	};

	class VBObject : public OpenGLObject {
	public: VBObject() : OpenGLObject(BuffDeleter) { glGenBuffers(1, resource.get()); }
	};

	class FBObject : public OpenGLObject {
	public: FBObject() : OpenGLObject(FrameBufferDeleter) { glGenFramebuffers(1, resource.get()); }
	};

	class ProgramObject : public OpenGLObject {
	public: ProgramObject() : OpenGLObject(ProgramDeleter) { *resource = glCreateProgram(); }
	};

	class RBObject : public OpenGLObject {
	public: RBObject() : OpenGLObject(RenderBufferDeleter) { glGenRenderbuffers(1, resource.get()); }
	};

	class ShaderObject : public OpenGLObject {
	public: ShaderObject(GLenum shaderType) : OpenGLObject(ShaderDeleter) { *resource = glCreateShader(shaderType); }
	};

	class TextureObject : public OpenGLObject {
	public: TextureObject() : OpenGLObject(TexDeleter) { glGenTextures(1, resource.get()); }
	};

	class VAObject : public OpenGLObject {
	public:	VAObject() : OpenGLObject(VAODeleter) { glGenVertexArrays(1, resource.get()); }
	};

	class VBObject_CUDA : public VBObject{
	public:
		cudaGraphicsResource* cuda;
		size_t size;
		uint32_t flags;
		VBObject_CUDA(size_t size, uint32_t flags) : VBObject(), size(size), flags(flags){
			glBindBuffer(GL_ARRAY_BUFFER, *resource);
			glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
			
			cudaErrorCheck(cudaGraphicsGLRegisterBuffer(&cuda, *resource, flags));
			cudaErrorCheck(cudaGraphicsMapResources(1, &cuda, 0));
		}

		~VBObject_CUDA() {
			cudaErrorCheck(cudaGraphicsUnmapResources(1, &cuda, 0));
			cudaErrorCheck(cudaGraphicsUnregisterResource(cuda));
		}
	};
}