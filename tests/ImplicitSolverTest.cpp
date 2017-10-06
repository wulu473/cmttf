
#include <boost/test/unit_test.hpp>

#include "ImplicitSolver.hpp"

#include "Attributes.hpp"
#include "System.hpp"
#include "Modules.hpp"
#include "Flat.hpp"


BOOST_AUTO_TEST_SUITE(ImplicitSolverTests)

BOOST_AUTO_TEST_CASE(setupFunctionRand)
{
  std::shared_ptr<System> sys = std::make_shared<System>(); 
  sys->initialise(1.,1.1,1.2,111,0.12,[](real,real){return 1.23;},[](real,real){return 4.;});
  Modules::addModule(sys);

  std::shared_ptr<Flat> flat = std::make_shared<Flat>();
  flat->initialise(5,0.,0.5);
  Modules::addModule(flat);

  ImplicitSolver::Vector states_old(10);
  states_old <<  7.55208555e-05,4.52495383e-01,
                 1.00000000e-06,0.00000000e+00,
                 2.73342916e-05,1.36632144e-01,
                 5.62725541e-05,5.17095798e-01,
                 9.07299124e-05,3.76788153e-01;

  ImplicitSolver::Vector states_new(10);

  states_new <<  1.00000000e-06,3.00000000e-01,
                 2.25208555e-05,1.52495383e-01,
                 3.07299124e-05,3.76788153e-01,
                 9.07299124e-05,3.76788153e-01,
                 5.62725541e-05,5.17095798e-01;


  std::shared_ptr<ImplicitSolver> solver = std::make_shared<ImplicitSolver>();
  solver->initialise(1.0);


  ImplicitSolver::Vector f(10);
  f.fill(1.0);

  solver->function(states_old,states_new,0.2,0.,f);

  BOOST_CHECK_CLOSE(f[0],-8.00071387e-01 , 1e-5);
  BOOST_CHECK_CLOSE(f[1], 1.34563360e+11 , 1e-5);
  BOOST_CHECK_CLOSE(f[2],-7.99967200e-01 , 1e-5);
  BOOST_CHECK_CLOSE(f[3], 1.34850918e+08 , 1e-5);
  BOOST_CHECK_CLOSE(f[4],-7.99965853e-01 , 1e-5);
  BOOST_CHECK_CLOSE(f[5], 1.78961344e+08 , 1e-5);
  BOOST_CHECK_CLOSE(f[6],-7.99948023e-01 , 1e-5);
  BOOST_CHECK_CLOSE(f[7], 2.05276080e+07 , 1e-5);
  BOOST_CHECK_CLOSE(f[8],-8.00039545e-01 , 1e-5);
  BOOST_CHECK_CLOSE(f[9], 7.32411535e+07 , 1e-5);

  Modules::clear();
}

BOOST_AUTO_TEST_CASE(setupJacobianRand)
{
  std::shared_ptr<System> sys = std::make_shared<System>(); 
  sys->initialise(1.,1.1,1.2,111,0.12,[](real,real){return 1.23;},[](real,real){return 4.;});
  Modules::addModule(sys);

  std::shared_ptr<Flat> flat = std::make_shared<Flat>();
  flat->initialise(5,0.,0.5);
  Modules::addModule(flat);

  ImplicitSolver::Vector states(10);

  states<<  1.00000000e-06,3.00000000e-01,
                 2.25208555e-05,1.52495383e-01,
                 3.07299124e-05,3.76788153e-01,
                 9.07299124e-05,3.76788153e-01,
                 5.62725541e-05,5.17095798e-01;

  std::shared_ptr<ImplicitSolver> solver = std::make_shared<ImplicitSolver>();
  solver->initialise(1.0);

  ImplicitSolver::SpMatRowMaj J(10,10);

  solver->jacobian(states,0.2,0.,J);

  BOOST_CHECK_CLOSE(J.coeff(0,0),  1.00000000e+00, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(0,1),  0.00000000e+00, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(0,2),  1.52495383e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(0,3),  2.25208555e-05, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(1,0), -2.69126997e+17, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(1,1),  4.48545456e+11, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(1,2), -9.55949061e+06, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(1,3),  6.13235908e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(1,4),  1.92157421e+06, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(1,5), -1.09594115e+03, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(2,0), -3.00000000e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(2,1), -1.00000000e-06, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(2,2),  1.00000000e+00, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(2,3),  0.00000000e+00, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(2,4),  3.76788153e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(2,5),  3.07299124e-05, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(3,0), -7.01350516e+04, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(3,1), -1.68773321e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(3,2), -1.19761922e+13, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(3,3),  8.84375506e+08, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(3,4), -2.42843976e+05, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(3,5),  2.36498606e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(3,6),  1.42680826e+04, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(3,7), -2.40411833e+01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(4,2), -1.52495383e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(4,3), -2.25208555e-05, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(4,4),  1.00000000e+00, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(4,5),  0.00000000e+00, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(4,6),  3.76788153e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(4,7),  9.07299124e-05, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(5,0),  8.25898894e+05, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(5,1), -1.92515133e+01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(5,2), -1.26958952e+04, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(5,3), -5.49640557e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(5,4), -1.16476635e+13, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(5,5),  4.74989385e+08, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(5,6), -2.17290728e+05, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(5,7),  6.30503619e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(5,8),  1.09627652e+05, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(5,9), -8.34344173e+01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(6,4), -3.76788153e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(6,5), -3.07299124e-05, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(6,6),  1.00000000e+00, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(6,7),  0.00000000e+00, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(6,8),  5.17095798e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(6,9),  5.62725541e-05, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(7,2),  4.50540669e+03, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(7,3), -4.64366010e+00, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(7,4),  3.72541592e+04, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(7,5), -5.09401615e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(7,6), -4.52532779e+11, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(7,7),  5.44885925e+07, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(7,8), -4.53292218e+04, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(7,9),  5.40679302e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(8,6), -3.76788153e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(8,7), -9.07299124e-05, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(8,8),  1.00000000e+00, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(8,9),  0.00000000e+00, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(9,4),  7.16545735e+04, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(9,5), -4.08618079e+01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(9,6),  4.87368172e+04, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(9,7), -8.15233327e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(9,8), -2.60317330e+12, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(9,9),  1.41648956e+08, 1e-5);

  Modules::clear();
}
BOOST_AUTO_TEST_SUITE_END()
