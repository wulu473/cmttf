

#include "Modules.hpp"

#include "System.hpp"

std::list<std::shared_ptr<ModuleBase> > Modules::m_modules = ModulesListInitialiser::makeList<ModuleBase>();

//! Read settings file and create instances of active modules
void Modules::initialiseFromFile()
{
  std::list<std::string> activeModuleNames = Parameters::allActiveModules();

  for( auto moduleName : activeModuleNames)
  {
    addModule(Factory::createInitialised<ParameterisedModuleBase>(moduleName));
  }

  // Initialise system
  std::shared_ptr<System> system = std::make_shared<System>();
  system->initialiseFromFile();
  addModule(system);
}


void Modules::addModule(const std::shared_ptr<ModuleBase> module)
{
  m_modules.push_back(module);
}

void Modules::clear()
{
  m_modules.clear();
}
