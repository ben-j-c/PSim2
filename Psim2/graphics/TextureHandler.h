#pragma once
#include <GL/glew.h>
#include "Graphics.h"

class Texture;

struct TexUnit {
	std::string boundTextureName;
	std::shared_ptr<Texture> boundTexture;
	int index;
	struct TexUnit* next;
	struct TexUnit* prev;
};

class Texture {
public:
	std::string name;
	std::string fileName;
	Graphics::GLResource texHandle;
	std::unique_ptr<TexUnit, std::function<void(TexUnit*)>> boundUnit;
	int height, width, nChannels;
	bool renderTarget;

	Texture() {};
	Texture(const std::string& name, const std::string& fileName, bool flip = true);
	Texture(const std::string& name, int height, int width);
};

namespace TextureHandler {
	void init();
	std::shared_ptr<Texture> loadTexture(const std::string& fileName, const std::string& name);
	std::shared_ptr<Texture> getNamedTexture(const std::string& name);
	std::shared_ptr<Texture> makeTexture(const std::string& name, int width, int height);
	std::shared_ptr<Texture> useParameters(const std::string& name, GLenum param, GLint value);
	std::shared_ptr<Texture>& useParameters(std::shared_ptr<Texture>& tex, GLenum param, GLint value);
	int textureUnitCount();
	std::shared_ptr<Texture> bindToUnit(const std::string& texture, const std::string& unit);
	std::shared_ptr<Texture> bindToUnit(std::shared_ptr<Texture>& tex, const std::string& unit);
	std::shared_ptr<Texture> bindToUnit(const std::string& texture, uint32_t unit);
	void setSamplerUnit(const std::string& unitName, uint32_t unitNumber);
	GLuint getTextureHandle(const std::string& name);
};