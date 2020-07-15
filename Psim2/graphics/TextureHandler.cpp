#define STB_IMAGE_IMPLEMENTATION
#include "TextureHandler.h"
#include "GraphicsState.hpp"
#include "../stb_image.h"
#include <unordered_map>
#include <vector>

#define RESERVED_TEX_UNITS 10

using std::unordered_map;
using std::string;
using std::vector;

//texture name -> tex handle
static unordered_map<string, std::shared_ptr<Texture>> textures(1024);
//Unit name -> texture name
static unordered_map<string, std::shared_ptr<Texture>> mappedTexture(1024);
//texture name -> unit its mapped to
static unordered_map<string, int> texToUnit(1024);

/* To be used as a LRU queue for unit allocation
*/
class TexUnitLL{
public:
	std::vector<TexUnit> units;
	int size;
	TexUnit header;
	TexUnit footer;
	TexUnitLL() = default;
	TexUnitLL(int maxUnits) : size(maxUnits - RESERVED_TEX_UNITS) {
		units = std::vector<TexUnit>(maxUnits);

		header.next = &footer;
		header.prev = nullptr;
		footer.next = nullptr;
		footer.prev = &header;
		//First unit is reserved for CEGUI
		//Second unit is reserved for font rendering
		//Third unit is reserved for texture loading
		//Units 3-9 are for various purposes
		for (int i = RESERVED_TEX_UNITS; i < size ;i++) {
			units[i].boundTextureName = "";
			units[i].index = i;
			push_back(&units[i]);
		}
	}

	void push_back(TexUnit* u) {
		footer.prev->next = u;
		u->next = &footer;
		u->prev = footer.prev;
		footer.prev = u;
	}

	TexUnit* pop_front() {
		TexUnit* returner = header.next;
		header.next = header.next->next;
		header.next->prev = &header;
		return returner;
	}

	void remove(TexUnit* u) {
		u->prev->next = u->next;
		u->next->prev = u->prev;
	}

	void move_back(TexUnit* u) {
		remove(u);
		push_back(u);
	}

	TexUnit* operator[](int index) {
		return &units[index];
	}

	TexUnit& front() {
		return *header.next;
	}
};

TexUnitLL texUnitAllocator;

Texture::Texture(const string& name, const string& fileName, bool flip) :
	texHandle(new GLuint, Graphics::TexDeleter){
	this->name = name;
	this->fileName = fileName;
	stbi_set_flip_vertically_on_load(flip);
	uint8_t* data = stbi_load(fileName.data(), &width, &height, &nChannels, 4);
	if (!data) {
		throw std::runtime_error("data was nullptr\n");
	}
	glGenTextures(1, texHandle.get());

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, *texHandle);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	glGenerateMipmap(GL_TEXTURE_2D);
	stbi_image_free(data);
}

Texture::Texture(const string & name, int height, int width) :
	texHandle(new GLuint, Graphics::TexDeleter) {
	this->name = name;

	glGenTextures(1, texHandle.get());

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, *texHandle);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

static inline bool hasTexture(const string& name) {
	return textures.count(name) > 0;
}

void TextureHandler::init() {
	int maxUnits = textureUnitCount();
	texUnitAllocator = TexUnitLL(maxUnits);
}

std::shared_ptr<Texture> TextureHandler::loadTexture(const string& fileName, const string& name) {
	if (!hasTexture(name)) {
		auto tex = std::shared_ptr<Texture>(new Texture(name, fileName));
		textures[name] = tex;
		return tex;
	}
	else {
		return nullptr;
	}
}

std::shared_ptr<Texture> TextureHandler::getNamedTexture(const std::string & name) {
	if (textures.count(name)) {
		return textures[name];
	}
	else {
		throw std::runtime_error("No texture named \"" + name + "\"");
	}
	return nullptr;
}

std::shared_ptr<Texture> TextureHandler::makeTexture(const string & name, int width, int height) {
	if (!hasTexture(name)) {
		auto tex = std::shared_ptr<Texture>(new Texture(name, height, width));
		textures[name] = tex;
		return tex;
	}
	else {
		return false;
	}
}

std::shared_ptr<Texture> TextureHandler::useParameters(const string & texture, GLenum param, GLint value) {
	if (hasTexture(texture)) {
		auto tex = textures[texture];
		if (tex->boundUnit) {
			glActiveTexture(GL_TEXTURE0 + tex->boundUnit->index);
		}
		else {
			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_2D, *tex->texHandle);
		}
		glTexParameteri(GL_TEXTURE_2D, param, value);
	}
	else {
		return nullptr;
	}
	return textures[texture];
}

std::shared_ptr<Texture>& TextureHandler::useParameters(std::shared_ptr<Texture>& tex, GLenum param, GLint value) {
	if (tex->boundUnit) {
		glActiveTexture(GL_TEXTURE0 + tex->boundUnit->index);
		glTexParameteri(GL_TEXTURE_2D, param, value);
	}
	return tex;
}

int TextureHandler::textureUnitCount() {
	int returner;
	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &returner);
	return returner;
}

static void allocateUnit(std::shared_ptr<Texture>& tex) {
	TexUnit* u = texUnitAllocator.pop_front();//LRU texture unit
	texUnitAllocator.push_back(u);//u is now the most recently used
	//Unbind the texture from the unit in pointer logic
			//A unit can only have a single texture bound
	if (u->boundTexture) {
		tex->boundUnit = std::move(u->boundTexture->boundUnit);
	}
	else { //First allocation
		//Dont delete, just use unique to ensure proper usage
		tex->boundUnit = std::unique_ptr<TexUnit, std::function<void(TexUnit*)>>
			(u, [](TexUnit*) {; });
	}
	u->boundTextureName = tex->name; //Its newly bound texture
	u->boundTexture = tex;
	glActiveTexture(GL_TEXTURE0 + u->index);
	glBindTexture(GL_TEXTURE_2D, *tex->texHandle);
}

/*Bind the texture data to the named texture unit
*/
std::shared_ptr<Texture> TextureHandler::bindToUnit(const string & texture, const string & unit) {
	if (texUnitAllocator.header.next == nullptr) {
		init();
	}
	if (hasTexture(texture)) {
		auto tex = textures[texture];
		GLuint program = Graphics::program;
		GLint loc = glGetUniformLocation(program, unit.data());
		if (loc != -1) {
			//If the texture is unbound to a unit then bind the unit
			//then continue to set uniform
			if(!tex->boundUnit) {
				allocateUnit(tex);
			}
			else {
				texUnitAllocator.move_back(tex->boundUnit.get());
				glActiveTexture(GL_TEXTURE0 + tex->boundUnit->index);
			}
			glUniform1i(loc, tex->boundUnit->index);
			return tex;
		}
		else {
			throw std::runtime_error("Uniform \"" + unit + "\" does not exist.");
		}
	}
	return nullptr;
}

std::shared_ptr<Texture> TextureHandler::bindToUnit(std::shared_ptr<Texture>& tex, const std::string & unit) {
	GLuint program = Graphics::program;
	GLint loc = glGetUniformLocation(program, unit.data());
	if (loc == -1)
		throw std::runtime_error("Uniform \"" + unit + "\" does not exist.");

	if (tex->boundUnit) {
		glUniform1i(loc, tex->boundUnit->index);
	}
	else {
		allocateUnit(tex);
		glUniform1i(loc, tex->boundUnit->index);
	}

	return tex;
}

/* Dont use this for dynamically assigned texture units
*/
std::shared_ptr<Texture> TextureHandler::bindToUnit(const string & texture, uint32_t unit) {
	if (hasTexture(texture)) {
		auto tex = textures[texture];
		tex->boundUnit = nullptr;
		glActiveTexture(GL_TEXTURE0 + unit);
		glBindTexture(GL_TEXTURE_2D, *tex->texHandle);
		return tex;
	}
	return nullptr;
}

void TextureHandler::setSamplerUnit(const string & unitName, uint32_t unitNumber) {
	GLint loc = glGetUniformLocation(Graphics::program, unitName.c_str());
	if (loc != -1)
		glUniform1i(loc, unitNumber);
	else
		throw std::runtime_error("Uniform \"" + unitName + "\" does not exist.");

}

GLuint TextureHandler::getTextureHandle(const string & name) {
	if (hasTexture(name)) {
		return *textures[name]->texHandle;
	}
	else {
		return 0;
	}
}
