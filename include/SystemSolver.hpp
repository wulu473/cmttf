
#ifndef SYSTEMSOLVER_H_
#define SYSTEMSOLVER_H_

#include "ModuleBase.hpp"
#include "DataPatch.hpp"
#include "BoundaryCondition.hpp"

class SystemSolver : public ParameterisedModuleBase
{
  public:
    virtual std::string baseName() const final;
    SystemSolver();
    virtual ~SystemSolver();

    virtual void initialise(const real finalT, const real maxDt);
    virtual void initialiseFromParameters(const Parameters& params);

    virtual void advance(std::shared_ptr<DataPatch> data, const real t, const real dt, 
                 const unsigned int i) const;

    virtual real maxDt(std::shared_ptr<DataPatch> data, const real t) const;

    virtual real finalT() const;

    virtual int exitcode() const;

    virtual void setBoundaryConditions(std::shared_ptr<const BoundaryConditionContainer>);
  private:
    real m_finalT;
    real m_maxDt;

    std::shared_ptr<const BoundaryConditionContainer> m_bcs;
};

#endif
