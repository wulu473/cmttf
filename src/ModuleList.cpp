

#include "ModuleList.hpp"

#include "System.hpp"

std::list<std::shared_ptr<ModuleBase> > ModuleList::m_modules = ModulesListInitialiser::makeList<ModuleBase>();

//! Read settings file and create instances of active modules
void ModuleList::initialiseFromParameters(const Parameters& params)
{
  std::list<std::string> activeModuleNames = params.allActiveModules();

  for( auto moduleName : activeModuleNames)
  {
    std::shared_ptr<ParameterisedModuleBase> module = 
            Factory::create<ParameterisedModuleBase>(moduleName);
    module->initialiseFromParameters(params);
    BOOST_LOG_TRIVIAL(debug) << "Creating " << module->name();
    addModule(module);
  }

  // Initialise system
  std::shared_ptr<System> system = std::make_shared<System>();
  system->initialiseFromParameters(params);
  addModule(system);
}


void ModuleList::addModule(const std::shared_ptr<ModuleBase> module)
{
  m_modules.push_back(module);
}

void ModuleList::clear()
{
  m_modules.clear();
}
