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

  BOOST_CHECK_CLOSE(F[0],-3.99985450849366,1e-5);
  BOOST_CHECK_CLOSE(F[1],410072066.459576,1e-5);
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

  BOOST_CHECK_CLOSE(J(0,0), 0.0000000000000    ,1e-5);
  BOOST_CHECK_CLOSE(J(0,1), 0.0000000000000    ,1e-5);
  BOOST_CHECK_CLOSE(J(0,2), 0.0000000000000    ,1e-5);
  BOOST_CHECK_CLOSE(J(0,3),-5.0000000000000e-6 ,1e-5);
  BOOST_CHECK_CLOSE(J(0,4), 0.0000000000000    ,1e-5);
  BOOST_CHECK_CLOSE(J(0,5), 0.0000000000000    ,1e-5);
  BOOST_CHECK_CLOSE(J(0,6), 2.5854789900000    ,1e-5);
  BOOST_CHECK_CLOSE(J(0,7), 0.0002813627705    ,1e-5);
  BOOST_CHECK_CLOSE(J(0,8), 0.0000000000000    ,1e-5);
  BOOST_CHECK_CLOSE(J(0,9), 0.0000000000000    ,1e-5);

  BOOST_CHECK_CLOSE(J(1,0), 584.550870599868  ,1e-5);
  BOOST_CHECK_CLOSE(J(1,1),-0.629884688286771 ,1e-5);
  BOOST_CHECK_CLOSE(J(1,2),-133739.612311371  ,1e-5);
  BOOST_CHECK_CLOSE(J(1,3),-0.741794959985211 ,1e-5);
  BOOST_CHECK_CLOSE(J(1,4),-30006069448872.3  ,1e-5);
  BOOST_CHECK_CLOSE(J(1,5),3001655546.69776   ,1e-5);
  BOOST_CHECK_CLOSE(J(1,6),-1399711.51744954  ,1e-5); 
  BOOST_CHECK_CLOSE(J(1,7),1.10328597144009   ,1e-5);
  BOOST_CHECK_CLOSE(J(1,8),134072.172558479   ,1e-5);
  BOOST_CHECK_CLOSE(J(1,9),-225.840150000766  ,1e-5);

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
