
#ifndef MODULES
#define MODULES

#include <memory>

#include "Factory.hpp"
#include "ModuleBase.hpp"
#include <list>

namespace ModulesListInitialiser
{
  template<typename T>
  std::list<std::shared_ptr<T> > makeList() {std::list<std::shared_ptr<T> > t; return t;};
};

class ModuleList
{
  public:
    template<typename T>
    static std::shared_ptr<const T> module();

    //! Return pointer of active module. Guarantees there is only one active module of this kind
    template<typename T>
    static std::shared_ptr<const T> uniqueModule();

    template<typename T>
    static std::list<std::shared_ptr<const T> > modules();

    //! Return non-const pointers of active modules
    template<typename T>
    static std::list<std::shared_ptr<T> > mutableModules();

    static void addModule(const std::shared_ptr<ModuleBase>);

    //! Read parameter store and create instances of active modules
    static void initialiseFromParameters(const Parameters& params);

    static void clear();
  private:
    static std::list<std::shared_ptr<ModuleBase> > m_modules;
};

#include "ModuleListTemplates.cpp"

#endif
