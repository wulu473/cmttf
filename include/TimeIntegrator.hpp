#ifndef TIMEINTEGRATOR_H_
#define TIMEINTEGRATOR_H_

#include "ModuleBase.hpp"
#include "DataPatch.hpp"

class TimeIntegrator : public ParameterisedModuleBase
{
  public:
    virtual std::string name() const;
    virtual void initialise(); 

    void advance(std::shared_ptr<DataPatch> states, const real dt, const real t) const;

  protected:
    // alpha = 0.5 Crank Nicolson
    // alpha = 1.0 Euler backward
    // alpha = 0.0 Euler forward
    real m_alpha;
};

#endif
