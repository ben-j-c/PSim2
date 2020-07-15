#pragma once
#include "shapes/Shape.h"
#include <unordered_map>

namespace Shapes {
	typedef std::shared_ptr<Shape> SharedShape;

	extern std::unordered_map<std::string, SharedShape> map;
	SharedShape add(const std::string& name, Shape*);
	SharedShape remove(const std::string& name);
}