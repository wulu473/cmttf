#ifndef TIMEINTEGRATOR_H_
#define TIMEINTEGRATOR_H_

#include "ModuleBase.hpp"
#include "BoundaryCondition.hpp"
#include "DataPatch.hpp"

class TimeIntegrator : public ParameterisedModuleBase
{
  public:
    TimeIntegrator() {};
    ~TimeIntegrator() {};
    virtual std::string baseName() const final;
    virtual void initialise(); 

    virtual void advance(std::shared_ptr<DataPatch> states, const real dt, const real t) const;

    virtual void setBoundaryConditions(std::shared_ptr<const BoundaryConditionContainer>);
  protected:
    // alpha = 0.5 Crank Nicolson
    // alpha = 1.0 Euler backward
    // alpha = 0.0 Euler forward
    real m_alpha;

    std::shared_ptr<const BoundaryConditionContainer> m_bcs;
};

#endif
