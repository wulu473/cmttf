
#include "SystemSolver.hpp"
#include "Modules.hpp"
#include "TimeIntegrator.hpp"

SystemSolver::SystemSolver()
{

}

SystemSolver::~SystemSolver()
{

}

std::string SystemSolver::baseName() const
{
  return "Solver";
}

void SystemSolver::initialise(const real finalT, const real maxDt)
{
  m_finalT = finalT;
  m_maxDt = maxDt;
}

void SystemSolver::initialiseFromFile()
{
  initialise(
    getParameter<real>("finalT"),
    getParameter<real>("maxDt",std::numeric_limits<real>::max())
    );
}

void SystemSolver::advance(std::shared_ptr<DataPatch> data, const real t, const real dt, 
                 const unsigned int i) const
{
  Modules::uniqueModule<TimeIntegrator>()->advance(data,dt,t);
}

real SystemSolver::maxDt(std::shared_ptr<DataPatch> data, const real t) const
{
  return std::min(m_finalT-t,m_maxDt);
}

real SystemSolver::finalT() const
{
  return m_finalT;
}

int SystemSolver::exitcode() const
{
  return 0;
}
