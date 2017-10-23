
#include "RootFinder.hpp"
#include <boost/log/trivial.hpp>


std::string RootFinder::baseName() const
{
  return "RootFinder";
}

void RootFinder::solveSparse(const std::function<void(const EVector&, EVector&)>& /*f*/, 
        const std::function<void(const EVector&, ESpMatRowMaj&)>& /*J*/, EVector& /*x*/,
        const std::function<bool(EVector&)>& /*restrictDomain*/) const
{
  BOOST_LOG_TRIVIAL(error) << "RootFinder::solveSparse is called from the base class";
  throw InheritanceException();
}

bool RootFinder::allValid(EVector&)
{
  return true;
};
