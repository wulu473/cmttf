
#ifndef MODULES
#error ModulesTemplates.cpp must be included in Modules.hpp
#endif

template<typename T>
std::list<std::shared_ptr<T> > Modules::mutableModules()
{
  std::list<std::shared_ptr<T> > mods;
  for(std::list<std::shared_ptr<ModuleBase> >::const_iterator it = m_modules.begin(); it != m_modules.end(); ++it)
  {
    std::shared_ptr<T> mod = std::dynamic_pointer_cast<T>(*it);
    if(mod.use_count()!=0)
    {
      mods.push_back(mod);
    }
  }
  return mods;
}

//! Return a list of active modules of type T. May be empty if no modules are active
template<typename T>
std::list<std::shared_ptr<const T> > Modules::modules()
{
  std::list<std::shared_ptr<const T> > mods;
  for(std::list<std::shared_ptr<ModuleBase> >::const_iterator it = m_modules.begin(); it != m_modules.end(); ++it)
  {
    std::shared_ptr<T> mod = std::dynamic_pointer_cast<T>(*it);
    if(mod.use_count()!=0)
    {
      mods.push_back(mod);
    }
  }
  return mods;
}

template<typename T>
std::shared_ptr<const T> Modules::module()
{
  std::list<std::shared_ptr<const T> > mods = modules<T>();
  if(mods.size()>1)
  {
    BOOST_LOG_TRIVIAL(error) << "Multiple active modules of " << mods.front()->moduleName() << " found.";
    exit(3);
    return NULL;
  }
  else if(mods.size() == 0)
  {
    // TODO return empty shared ptr
    return NULL;
  }
  return mods.front();
}

template<typename T>
std::shared_ptr<const T> Modules::uniqueModule()
{
  std::list<std::shared_ptr<const T> > mods = modules<T>();
  if(mods.size()>1)
  {
    BOOST_LOG_TRIVIAL(error) << "Multiple active modules of type " << mods.front()->moduleName() << " found.";
    exit(3);
    return NULL;
  }
  else if(mods.size() == 0)
  {
    BOOST_LOG_TRIVIAL(error) << "No active modules of type " << std::make_shared<T>()->moduleName() << " found.";
    exit(3);
    return NULL;
  }
  return mods.front();
}

