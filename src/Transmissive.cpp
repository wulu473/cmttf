
#include "Transmissive.hpp"

REGISTERIMPL(Transmissive);

std::string Transmissive::moduleName() const
{
  return "Transmissive";
}

State Transmissive::ghostState(const State& domainState, const State&) const
{
  return domainState;
}

