
#ifndef SYSTEMSOLVER_H_
#define SYSTEMSOLVER_H_

#include "ModuleBase.hpp"
#include "DataPatch.hpp"

class SystemSolver : public ParameterisedModuleBase
{
  public:
    virtual std::string baseName() const final;
    SystemSolver();
    virtual ~SystemSolver();

    virtual void initialise(const real finalT, const real maxDt);
    virtual void initialiseFromFile();

    virtual void advance(std::shared_ptr<DataPatch> data, const real t, const real dt, 
                 const unsigned int i) const;

    virtual real maxDt(std::shared_ptr<DataPatch> data, const real t) const;

    virtual real finalT() const;

    virtual int exitcode() const;
  private:
    real m_finalT;
    real m_maxDt;
};

#endif
