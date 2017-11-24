

#include "SystemBase.hpp"

#include "SystemAttributes.hpp"

std::string SystemBase::baseName() const
{
  return "System";
}
    
std::string SystemBase::moduleName() const
{
  return "";
}

int SystemBase::stencilSize() const
{
  return SystemAttributes::stencilSize;
}

bool SystemBase::checkValid(Ref<State> /*state*/) const
{
  // default is that every state is valid TODO Consider checking for nans
  return true;
}

DerivedVariablesMap SystemBase::derivedVariables() const
{
  // Assume no derived variables
  return DerivedVariablesMap();
}

//! Return a factor the time derivative is scaled with
State SystemBase::factorTimeDeriv() const
{
  State factor;
  factor.fill(1.);
  return factor;
}

