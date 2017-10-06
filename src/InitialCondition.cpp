
#include "InitialCondition.hpp"

std::string InitialCondition::name() const
{
  return "InitialCondition";
}

InitialCondition::InitialCondition()
{

}

InitialCondition::~InitialCondition()
{

}

void InitialCondition::setData(std::shared_ptr<DataPatch>) const
{
  BOOST_LOG_TRIVIAL(error) << "InitialCondition::setData called from base class.";
  throw InheritanceException();
}
