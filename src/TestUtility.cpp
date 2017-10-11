
#include <boost/test/unit_test.hpp>

#include "TestUtility.hpp"
#include "System.hpp"

bool TestUtility::SystemTest::checkJacobianNumerically(std::shared_ptr<System> sys, const StencilArray& u,
    const real dx, const real x, const real t, const real eps)
{
  const State nullState = 0.*u[0];
  StencilArray u0;
  u0.fill(nullState);
  const StencilJacobian J = sys->J(u, dx, x, t);
  const State F = sys->F(u, dx, x, t);
  for(unsigned int i=0; i<2*SystemAttributes::stencilSize+1; i++)
  { 
    for(unsigned int j=0; j<SystemAttributes::stateSize; j++)
    {
      for(unsigned int k=0; k<SystemAttributes::stateSize; k++)
      {
        const unsigned int s = i*SystemAttributes::stateSize + j;
        StateArray du = u0;
        du[i][j] += eps;
        State dF = (sys->F(u+du, dx, x, t) - F)/eps;
        BOOST_CHECK_CLOSE(J(k,s), dF[k],0.5); // Check for 0.5% accuracy
      }
    }
  }
  return true;
}
