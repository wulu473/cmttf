
#include "SystemSolver.hpp"
#include "ModuleList.hpp"
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

void SystemSolver::initialise(const real TFin, const real maxDt)
{
  m_finalT = TFin;
  m_maxDt = maxDt;
}

void SystemSolver::initialiseFromParameters(const Parameters& params)
{
  initialise(
    getParameter<real>(params, "finalT"),
    getParameter<real>(params, "maxDt",std::numeric_limits<real>::max())
    );
}

void SystemSolver::advance(std::shared_ptr<DataPatch> data, const real t, const real dt, 
                 const unsigned int /*iter*/) const
{
  ModuleList::uniqueModule<TimeIntegrator>()->advance(data,dt,t);
}

real SystemSolver::maxDt(std::shared_ptr<DataPatch> /*data*/, const real t) const
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

void SystemSolver::setBoundaryConditions(std::shared_ptr<const BoundaryConditionContainer> bcs)
{
  m_bcs = bcs;
  ModuleList::uniqueModule<TimeIntegrator>(); // Make sure only one TimeIntegrator is active
  ModuleList::mutableModules<TimeIntegrator>().front()->setBoundaryConditions(bcs);
}
