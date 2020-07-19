#include "SettingsWrapper.h"


std::unique_ptr<SettingsWrapper> SettingsWrapper::singleton = std::unique_ptr<SettingsWrapper>(new SettingsWrapper());