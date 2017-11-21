
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
    void readFile(const std::string&);

    template<typename T>
    const T getParameter(const std::string&) const;

    template<typename T>
    const T getParameter(const std::string&, const T&) const;

    template<typename T>
    const std::vector<T> getVectorParameter(const std::string&) const;

    const TimeSpaceDependReal getExpressionParameter(const std::string&) const;
    const std::vector<TimeSpaceDependReal> getExpressionVectorParameter(const std::string&) const;

    bool exists(const std::string&) const ; 
    int getLength(const std::string&) const;

    template<typename T>
    void deleteUnactive(std::list<T*>&);

    std::string activeModule(const std::string&) const;
    std::list<std::string> activeModules(const std::string&) const;
    std::list<std::string> allActiveModules() const;

  private:
    libconfig::Config m_cfg;

    template<typename T>
    const T convertScalar(const std::string&) const;

    template<typename T>
    const T convertArray(const std::string&) const;
};

#include "ParametersTemplates.cpp"
#endif


