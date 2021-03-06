#ifndef MODULEBASE_H_
#error ModuleBaseTemplates.cpp must be included in ModuleBase.hpp
#endif

#include <sstream>
#include "TimeSpaceDependReal.hpp"

template<typename T>
const T ParameterisedModuleBase::getParameter(const Parameters& params, const std::string& variableName) const
{
  std::ostringstream n;
  n << name() << "." << variableName;
  return params.getParameter<T>(n.str());
}

/** Specialisation for States.
 *
 * TODO Figure out why this does not work without the 'inline'
*/
template<>
inline const State
ParameterisedModuleBase::getParameter<State>(const Parameters& params, const std::string& variableName) const
{
  std::vector<real> stateVec = getVectorParameter<real>(params, variableName);
  State state;
  if(((unsigned int) state.size()) != stateVec.size())
  {
    std::ostringstream n;
    n << name() << "." << variableName;
    BOOST_LOG_TRIVIAL(error) << "Length of state vector in " << n.str() << " does not match requirements of the system.";
    exit(20);
  }
  for(unsigned int i=0;i<stateVec.size();i++)
  {
    state[i] = stateVec[i];
  }
  return state;
}

/** Specialisation for Expressions
 *
 * TODO Figure out why this does not work without the 'inline'
*/
template<>
inline const TimeSpaceDependReal
ParameterisedModuleBase::getParameter<TimeSpaceDependReal>(const Parameters& params, const std::string& variableName) const
{
  std::ostringstream n;
  n << name() << "." << variableName;
  return params.getExpressionParameter(n.str());
}

template<typename T>
const T ParameterisedModuleBase::getParameter(const Parameters& params, const std::string& variableName, const T& defaultValue) const
{
  std::ostringstream n;
  n << name() << "." << variableName;
  return params.getParameter<T>(n.str(),defaultValue);
}

/** Specialisation for Expressions
 *
 * TODO Figure out why this does not work without the 'inline'
*/
template<>
inline const std::vector<TimeSpaceDependReal> ParameterisedModuleBase::getVectorParameter(
            const Parameters& params, const std::string& variableName) const
{
  std::ostringstream n;
  n << name() << "." << variableName;
  return params.getExpressionVectorParameter(n.str());
}

template<typename T>
const std::vector<T> ParameterisedModuleBase::getVectorParameter(const Parameters& params, const std::string& variableName) const
{
  std::ostringstream n;
  n << name() << "." << variableName;
  return params.getVectorParameter<T>(n.str());
}
