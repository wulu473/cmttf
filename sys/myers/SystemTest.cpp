#include <boost/test/unit_test.hpp>

#include "System.hpp"
#include "TestUtility.hpp"

BOOST_AUTO_TEST_SUITE(SystemTest)

BOOST_AUTO_TEST_CASE(checkValidTest)
{
  std::shared_ptr<System> sys = std::make_shared<System>(); 
  State valid, invalid;
  valid << 1.;
  invalid << -1.;

  // Check a valid state is the same
  BOOST_CHECK(sys->checkValid(valid));
  BOOST_CHECK_CLOSE(valid[0], 1., 1e-10);

  // Check if an invalid state is corrected to a positive one
  BOOST_CHECK(sys->checkValid(invalid));
  BOOST_CHECK(invalid[0] > 0.);
}

BOOST_AUTO_TEST_CASE(J_num)
{
  const real dx = 0.1;
  const real x = 0.;
  const real t = 0.;
  std::shared_ptr<System> sys = std::make_shared<System>(); 
  sys->initialise(/*mu*/1.,
                  /*sigma*/0.12,
                  /*g1*/1.,
                  /*g2*/2.,
                  /*tau*/TimeSpaceDependReal("1.23"),
                  /*beta*/TimeSpaceDependReal("4."));

  State state1, state2, state3, state4, state5;
  state1 << 7.55208555;
  state2 << 1.00000000;
  state3 << 2.73342916;
  state4 << 5.62725541;
  state5 << 9.07299124;
  StencilArray u;
  u << state1,state2,state3,state4,state5;

  TestUtility::SystemTest::checkJacobianNumerically(sys,u,dx,x,t,1e-4);
}

BOOST_AUTO_TEST_CASE(F_old_new)
{
  // if states_new == states_old F_lin == F

  const real dx = 0.1;
  const real x = 0.;
  const real t = 0.;
  std::shared_ptr<System> sys = std::make_shared<System>(); 
  sys->initialise(/*mu*/1.,
                  /*sigma*/0.12,
                  /*g1*/0.,
                  /*g2*/0.,
                  /*tau*/TimeSpaceDependReal("1.23"),
                  /*beta*/TimeSpaceDependReal("4."));

  State state1, state2, state3, state4, state5;
  state1 << 7.55208555;
  state2 << 1.00000000;
  state3 << 2.73342916;
  state4 << 5.62725541;
  state5 << 9.07299124;
  StencilArray u;
  u << state1,state2,state3,state4,state5;

  const State F_lin = sys->FLinear(u,u,dx,x,t);
  const State F = sys->F(u,dx,x,t);

  BOOST_CHECK_CLOSE(F_lin[0], F[0], 1e-10);
}

BOOST_AUTO_TEST_SUITE_END()
