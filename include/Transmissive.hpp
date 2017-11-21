
#ifndef TRANSMISSIVE_H_
#define TRANSMISSIVE_H_

#include "BoundaryCondition.hpp"

class Transmissive : public BoundaryCondition
{
  REGISTER(Transmissive)
  public:
    virtual std::string moduleName() const;
    virtual State ghostState(const State&, const State&) const;
};

#endif
