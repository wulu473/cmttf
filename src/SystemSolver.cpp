
#include "SystemSolver.hpp"

SystemSolver::SystemSolver()
{

}

SystemSolver::~SystemSolver()
{

}

std::string SystemSolver::name() const
{
  return "Solver";
}

void SystemSolver::initialise(const real finalT, const real maxDt)
{
  m_finalT = finalT;
  m_maxDt = maxDt;

  m_implicitSolver = std::make_shared<ImplicitSolver>();
  m_implicitSolver->initialise(1.0);
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
  m_implicitSolver->advance(data,dt,t);
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
