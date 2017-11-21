
#include "BoundaryCondition.hpp"
#include "Factory.hpp"
#include "ImplicitSolver.hpp"

void BoundaryCondition::setOrientation(const Orientation o)
{
  m_orientation = o;
}

std::string BoundaryCondition::baseName() const
{
  if(m_orientation == Left)
  {
    return "BoundaryConditions.Left";
  }
  else
  {
    return "BoundaryConditions.Right";
  }
}

//! Return the ghost state
/**
 *
 * [in] domainState Corresponding state inside the domain
 * [in] peridoicDomainState Corresponding state on the other side of the domain
 */
State BoundaryCondition::ghostState(const State& /*domainState*/,
            const State& /*periodicDomainState*/) const
{
  BOOST_LOG_TRIVIAL(error) << "BoundaryCondition::ghostState is called from the base class";
  throw InheritanceException();
  return State();
}

BoundaryConditionContainer::BoundaryConditionContainer()
{

}

void BoundaryConditionContainer::initialiseFromParameters(const Parameters& params)
{
  std::string lStr = params.activeModule("BoundaryConditions.Left"); 
  std::string rStr = params.activeModule("BoundaryConditions.Right"); 

  m_left = Factory::create<BoundaryCondition>(lStr);
  m_right = Factory::create<BoundaryCondition>(rStr);

  m_left->setOrientation(BoundaryCondition::Left);
  m_right->setOrientation(BoundaryCondition::Right);
}



