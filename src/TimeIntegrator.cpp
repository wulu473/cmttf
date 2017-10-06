
#include "TimeIntegrator.hpp"

std::string TimeIntegrator::name() const
{
  return "TimeIntegrator";
}

void TimeIntegrator::initialise()
{

}

void TimeIntegrator::advance(std::shared_ptr<DataPatch> /*states*/, const real /*dt*/, 
          const real /*t*/) const
{
  BOOST_LOG_TRIVIAL(error) << "TimeIntegrator::advance called from base class";
  throw InheritanceException();
}

