

#include "ModuleBase.hpp"

std::string ModuleBase::name() const
{
  std::string base = this->baseName();
  std::string module = this->moduleName();
  if (base.compare("") == 0)
  {
    return module;
  }
  else if (module.compare("") == 0)
  {
    return base;
  }
  else
  {
    return this->baseName() + "." + this->moduleName();
  }
}

ModuleBase::~ModuleBase()
{

}

void ParameterisedModuleBase::initialiseFromParameters(const Parameters& /*params*/)
{
  BOOST_LOG_TRIVIAL(error) << this->name() << " does not require parameters. Consider inheriting from ModuleBase instead";
  throw InheritanceException();
}

std::string ModuleBase::baseName() const
{
  return "";
}
std::string ModuleBase::moduleName() const
{
  return "";
}

