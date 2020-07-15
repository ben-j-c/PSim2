#include "ShapeStore.h"

std::unordered_map<std::string, std::shared_ptr<Shape>> Shapes::map;

std::shared_ptr<Shape> Shapes::add(const std::string & name, Shape * shape) {
	auto retVal = map.emplace(name, shape);
	if (retVal.second) {
		return retVal.first->second;
	}
	return nullptr;
}

std::shared_ptr<Shape> Shapes::remove(const std::string & name) {
	if (map.count(name) == 0)
		return nullptr;
	auto retVal = map[name];
	map.erase(name);
	return retVal;
}
