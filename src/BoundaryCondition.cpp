
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

std::shared_ptr<const BoundaryCondition> BoundaryConditionContainer::left() const
{
  return m_left;
}

std::shared_ptr<const BoundaryCondition> BoundaryConditionContainer::right() const
{
  return m_right;
}

void BoundaryConditionContainer::initialise(const std::shared_ptr<BoundaryCondition> left,
                                            const std::shared_ptr<BoundaryCondition> right)
{
  m_left = left;
  m_right = right;
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



