
#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <libconfig.h++>
#include <string>
#include <list>
#include <vector>

#include "Attributes.hpp"
#include "TimeSpaceDependReal.hpp"

class Parameters
{
  public:
    static void readFile(const std::string&);

    template<typename T>
    static const T getParameter(const std::string&);

    template<typename T>
    static const T getParameter(const std::string&, const T&);

    template<typename T>
    static const std::vector<T> getVectorParameter(const std::string&);

    static const TimeSpaceDependReal getExpressionParameter(const std::string&);

    static bool exists(const std::string&); 
    static int getLength(const std::string&);

    template<typename T>
    static void deleteUnactive(std::list<T*>&);

    static std::string activeModule(const std::string&);
    static std::list<std::string> activeModules(const std::string&);
    static std::list<std::string> allActiveModules();

  private:
    static libconfig::Config m_cfg;

    template<typename T>
    static const T convertScalar(const std::string&);

    template<typename T>
    static const T convertArray(const std::string&);
};

#include "ParametersTemplates.cpp"
#endif


