
#ifndef PERIODIC_H_
#define PERIODIC_H_

#include "BoundaryCondition.hpp"

class Periodic : public BoundaryCondition
{
  REGISTER(Periodic)
  public:
    virtual std::string moduleName() const;
    virtual State ghostState(const State&, const State&) const;
};

#endif
