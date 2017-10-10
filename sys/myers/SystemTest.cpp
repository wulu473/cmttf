#include <boost/test/unit_test.hpp>

#include "System.hpp"
#include "SystemTestUtility.hpp"

BOOST_AUTO_TEST_SUITE(SystemTest)


BOOST_AUTO_TEST_CASE(J_num)
{
  const real dx = 0.1;
  const real x = 0.;
  const real t = 0.;
  std::shared_ptr<System> sys = std::make_shared<System>(); 
  sys->initialise(/*mu*/1.,
                  /*sigma*/0.12,
                  /*tau*/[](real,real){return 1.23;},
                  /*beta*/[](real,real){return 4.;});

  State state1, state2, state3, state4, state5;
  state1 << 7.55208555;
  state2 << 1.00000000;
  state3 << 2.73342916;
  state4 << 5.62725541;
  state5 << 9.07299124;
  StencilArray u;
  u << state1,state2,state3,state4,state5;

  SystemTestUtility::checkJacobianNumerically(sys,u,dx,x,t,1e-4);
}

BOOST_AUTO_TEST_SUITE_END()
