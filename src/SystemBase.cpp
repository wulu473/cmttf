

#include "SystemBase.hpp"

#include "SystemAttributes.hpp"

std::string SystemBase::name() const
{
  return "System";
}

int SystemBase::stencilSize() const
{
  return SystemAttributes::stencilSize;
}

