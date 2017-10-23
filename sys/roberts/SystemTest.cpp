#include <boost/test/unit_test.hpp>

#include "System.hpp"
#include "TestUtility.hpp"

BOOST_AUTO_TEST_SUITE(SystemTest)

BOOST_AUTO_TEST_CASE(FTest)
{
  std::shared_ptr<System> sys = std::make_shared<System>(); 
  sys->initialise(1.,1.1,1.2,111,0.12,TimeSpaceDependReal("1.23"),TimeSpaceDependReal("4"));

  State state1, state2, state3, state4, state5;
  state1 << 7.55208555e-05,4.52495383e-01;
  state2 << 1.00000000e-06,0.00000000e+00;
  state3 << 2.73342916e-05,1.36632144e-01;
  state4 << 5.62725541e-05,5.17095798e-01;
  state5 << 9.07299124e-05,3.76788153e-01;

  StencilArray states;
  states << state1,state2,state3,state4,state5;

  State F = sys->F(states,0.1,0.,0.);

  BOOST_CHECK_CLOSE(F[0],-3.99985450849366,1e-10);
  BOOST_CHECK_CLOSE(F[1],451079273.10553366,1e-10);
}

BOOST_AUTO_TEST_CASE(JTest)
{
  std::shared_ptr<System> sys = std::make_shared<System>(); 
  sys->initialise(/*mu*/1.,
                  /*rho*/1.1,
                  /*g1*/1.2,
                  /*g2*/111,
                  /*tau*/0.12,
                  /*sigma*/TimeSpaceDependReal("1.23"),
                  /*beta*/TimeSpaceDependReal("4."));

  State state1, state2, state3, state4, state5;
  state1 << 7.55208555e-05,4.52495383e-01;
  state2 << 1.00000000e-06,0.00000000e+00;
  state3 << 2.73342916e-05,1.36632144e-01;
  state4 << 5.62725541e-05,5.17095798e-01;
  state5 << 9.07299124e-05,3.76788153e-01;

  StencilArray states;
  states << state1,state2,state3,state4,state5;

  StencilJacobian J = sys->J(states,0.1,0.,0.);

  BOOST_CHECK_CLOSE(J(0,0), 0.0000000000000    ,1e-10);
  BOOST_CHECK_CLOSE(J(0,1), 0.0000000000000    ,1e-10);
  BOOST_CHECK_CLOSE(J(0,2), 0.0000000000000    ,1e-10);
  BOOST_CHECK_CLOSE(J(0,3),-5.0000000000000e-6 ,1e-10);
  BOOST_CHECK_CLOSE(J(0,4), 0.0000000000000    ,1e-10);
  BOOST_CHECK_CLOSE(J(0,5), 0.0000000000000    ,1e-10);
  BOOST_CHECK_CLOSE(J(0,6), 2.5854789900000    ,1e-10);
  BOOST_CHECK_CLOSE(J(0,7), 0.0002813627705    ,1e-10);
  BOOST_CHECK_CLOSE(J(0,8), 0.0000000000000    ,1e-10);
  BOOST_CHECK_CLOSE(J(0,9), 0.0000000000000    ,1e-10);

  BOOST_CHECK_CLOSE(J(1,0), 643.0059576598547 ,1e-10);
  BOOST_CHECK_CLOSE(J(1,1),-0.6928731571154482,1e-10);
  BOOST_CHECK_CLOSE(J(1,2),-147113.5735425081 ,1e-10);
  BOOST_CHECK_CLOSE(J(1,3),-0.8159744559837322,1e-10);
  BOOST_CHECK_CLOSE(J(1,4),-33006676393759.535,1e-10);
  BOOST_CHECK_CLOSE(J(1,5), 3301821101.3675365,1e-10);
  BOOST_CHECK_CLOSE(J(1,6),-1539682.6691944941,1e-10); 
  BOOST_CHECK_CLOSE(J(1,7), 1.213614568584099 ,1e-10);
  BOOST_CHECK_CLOSE(J(1,8), 147479.38981432692,1e-10);
  BOOST_CHECK_CLOSE(J(1,9),-248.4241650008426 ,1e-10);

}

BOOST_AUTO_TEST_CASE(J_num)
{
  const real dx = 0.1;
  const real x = 0.;
  const real t = 0.;
  std::shared_ptr<System> sys = std::make_shared<System>(); 
  sys->initialise(/*mu*/1.,
                  /*rho*/1.1,
                  /*g1*/1.2,
                  /*g2*/111,
                  /*tau*/0.12,
                  /*sigma*/TimeSpaceDependReal("1.23"),
                  /*beta*/TimeSpaceDependReal("4."));

  State state1, state2, state3, state4, state5;
  state1 << 7.55208555,4.52495383;
  state2 << 1.00000000,0.00000000;
  state3 << 2.73342916,1.36632144;
  state4 << 5.62725541,5.17095798;
  state5 << 9.07299124,3.76788153;
  StencilArray u;
  u << state1,state2,state3,state4,state5;

  TestUtility::SystemTest::checkJacobianNumerically(sys,u,dx,x,t,1e-10);
}

BOOST_AUTO_TEST_SUITE_END()
