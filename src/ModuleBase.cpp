

#include "ModuleBase.hpp"

std::string ModuleBase::name() const
{
  return "";
}

ModuleBase::~ModuleBase()
{

}

void ParameterisedModuleBase::initialiseFromFile()
{
  BOOST_LOG_TRIVIAL(error) << this->name() << " does not require parameters. Consider inheriting from ModuleBase instead";
  throw InheritanceException();
}

/** Extract the name of the child module
*/
std::string ModuleBase::moduleName() const
{
  std::string fullName = name();
  std::size_t pos = fullName.find_last_of(".");
  if (pos == std::string::npos)
  {
    pos = 0;
  }
  else 
  {
    pos++; // move to the right otherwise we include the '.'
  }
  return fullName.substr(pos);
}

