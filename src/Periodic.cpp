
#include "Periodic.hpp"

REGISTERIMPL(Periodic);

std::string Periodic::moduleName() const
{
  return "Periodic";
}

State Periodic::ghostState(const State&, const State& periodicDomainState) const
{
  return periodicDomainState;
}

