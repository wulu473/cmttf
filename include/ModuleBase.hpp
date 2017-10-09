
#ifndef MODULEBASE_H_
#define MODULEBASE_H_

#include <string>
#include <vector>

#include "Parameters.hpp"
#include "Factory.hpp"

class InheritanceException : public std::exception
{

};

class ModuleBase
{
  public:
    virtual std::string name() const final;
    virtual ~ModuleBase();
    virtual std::string baseName() const;
    virtual std::string moduleName() const;

  protected:
};

/**
  This class acts as an interface for other classes which need parameters from the settings file.
  Other classes should inherit from this one and parameters can be obtained by calling the function getParameter
  */
class ParameterisedModuleBase: public ModuleBase
{
  protected:
    template<typename T>
      const T getParameter(const std::string&) const;
    template<typename T>
      const T getParameter(const std::string&, const T&) const;
    template<typename T>
      const std::vector<T> getVectorParameter(const std::string&) const;
  public:
    virtual void initialiseFromFile();
};

#include "ModuleBaseTemplates.cpp"
#endif

