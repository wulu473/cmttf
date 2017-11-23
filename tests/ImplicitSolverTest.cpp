
#include <boost/test/unit_test.hpp>

#include "ImplicitSolver.hpp"

#include "Attributes.hpp"
#include "ModuleList.hpp"
#include "Flat.hpp"
#include "Transmissive.hpp"
#include "Periodic.hpp"

#include "System.hpp"

#ifndef ROBERTS
#warning ImplicitSolver unit tests only work for system roberts
#warning Skipping ImplicitSolverTest.cpp ...
#else

BOOST_AUTO_TEST_SUITE(ImplicitSolverTests)

BOOST_AUTO_TEST_CASE(setupFunctionRand)
{
  std::shared_ptr<System> sys = std::make_shared<System>(); 
  sys->initialise(1.,1.1,1.2,111,0.12,TimeSpaceDependReal("1.23"),TimeSpaceDependReal("4."));
  ModuleList::addModule(sys);

  std::shared_ptr<Flat> flat = std::make_shared<Flat>();
  flat->initialise(5,0.,0.5);
  ModuleList::addModule(flat);

  std::shared_ptr<BoundaryConditionContainer> bcs = std::make_shared<BoundaryConditionContainer>();
  bcs->initialise(std::make_shared<Transmissive>(), std::make_shared<Transmissive>());

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
  solver->setBoundaryConditions(bcs);


  ImplicitSolver::Vector f(10);
  f.fill(1.0);

  solver->function(states_old,states_new,0.2,0.,f);

  BOOST_CHECK_CLOSE(f[0],-0.80007138652901511, 1e-10);
  BOOST_CHECK_CLOSE(f[1], 148019696367.2915  , 1e-10);
  BOOST_CHECK_CLOSE(f[2],-0.79996720047756498, 1e-10);
  BOOST_CHECK_CLOSE(f[3], 148336010.18978    , 1e-10);
  BOOST_CHECK_CLOSE(f[4],-0.79996585274956999, 1e-10);
  BOOST_CHECK_CLOSE(f[5], 196857477.9412564  , 1e-10);
  BOOST_CHECK_CLOSE(f[6],-0.79994802300736723, 1e-10);
  BOOST_CHECK_CLOSE(f[7], 22580368.750865635 , 1e-10);
  BOOST_CHECK_CLOSE(f[8],-0.80003954501314734, 1e-10);
  BOOST_CHECK_CLOSE(f[9], 80565268.873657376 , 1e-10);

  ModuleList::clear();
}

BOOST_AUTO_TEST_CASE(setupFunctionRandPeriodicRight)
{
  std::shared_ptr<System> sys = std::make_shared<System>(); 
  sys->initialise(1.,1.1,1.2,111,0.12,TimeSpaceDependReal("1.23"),TimeSpaceDependReal("4."));
  ModuleList::addModule(sys);

  std::shared_ptr<Flat> flat = std::make_shared<Flat>();
  flat->initialise(5,0.,0.5);
  ModuleList::addModule(flat);

  std::shared_ptr<BoundaryConditionContainer> bcs = std::make_shared<BoundaryConditionContainer>();
  bcs->initialise(std::make_shared<Periodic>(), std::make_shared<Periodic>());

  // Check right boundary condition by cycling through the states and then only test the last flux
  ImplicitSolver::Vector states_old(10);
  states_old <<  5.62725541e-05,5.17095798e-01,
                 9.07299124e-05,3.76788153e-01,
                 7.55208555e-05,4.52495383e-01,
                 1.00000000e-06,0.00000000e+00,
                 2.73342916e-05,1.36632144e-01;

  ImplicitSolver::Vector states_new(10);

  states_new <<  9.07299124e-05,3.76788153e-01,
                 5.62725541e-05,5.17095798e-01,
                 1.00000000e-06,3.00000000e-01,
                 2.25208555e-05,1.52495383e-01,
                 3.07299124e-05,3.76788153e-01;

  std::shared_ptr<ImplicitSolver> solver = std::make_shared<ImplicitSolver>();
  solver->initialise(1.0);
  solver->setBoundaryConditions(bcs);


  ImplicitSolver::Vector f(10);
  f.fill(1.0);

  solver->function(states_old,states_new,0.2,0.,f);

  BOOST_CHECK_CLOSE(f[8],-0.79996585274956999, 1e-10);
  BOOST_CHECK_CLOSE(f[9], 196857477.9412564  , 1e-10);

  ModuleList::clear();
}

BOOST_AUTO_TEST_CASE(setupFunctionRandPeriodicLeft)
{
  std::shared_ptr<System> sys = std::make_shared<System>(); 
  sys->initialise(1.,1.1,1.2,111,0.12,TimeSpaceDependReal("1.23"),TimeSpaceDependReal("4."));
  ModuleList::addModule(sys);

  std::shared_ptr<Flat> flat = std::make_shared<Flat>();
  flat->initialise(5,0.,0.5);
  ModuleList::addModule(flat);

  std::shared_ptr<BoundaryConditionContainer> bcs = std::make_shared<BoundaryConditionContainer>();
  bcs->initialise(std::make_shared<Periodic>(), std::make_shared<Periodic>());

  // Check left boundary condition by cycling through the states and then only test the last flux
  ImplicitSolver::Vector states_old(10);
  states_old <<  2.73342916e-05,1.36632144e-01,
                 5.62725541e-05,5.17095798e-01,
                 9.07299124e-05,3.76788153e-01,
                 7.55208555e-05,4.52495383e-01,
                 1.00000000e-06,0.00000000e+00;  

  ImplicitSolver::Vector states_new(10);
  states_new <<  3.07299124e-05,3.76788153e-01,
                 9.07299124e-05,3.76788153e-01,
                 5.62725541e-05,5.17095798e-01,
                 1.00000000e-06,3.00000000e-01,
                 2.25208555e-05,1.52495383e-01;

  std::shared_ptr<ImplicitSolver> solver = std::make_shared<ImplicitSolver>();
  solver->initialise(1.0);
  solver->setBoundaryConditions(bcs);


  ImplicitSolver::Vector f(10);
  f.fill(1.0);

  solver->function(states_old,states_new,0.2,0.,f);

  BOOST_CHECK_CLOSE(f[0],-0.79996585274956999, 1e-10);
  BOOST_CHECK_CLOSE(f[1], 196857477.9412564  , 1e-10);

  ModuleList::clear();
}
BOOST_AUTO_TEST_CASE(setupJacobianRand)
{
  std::shared_ptr<System> sys = std::make_shared<System>(); 
  sys->initialise(1.,1.1,1.2,111,0.12,TimeSpaceDependReal("1.23"),TimeSpaceDependReal("4."));
  ModuleList::addModule(sys);

  std::shared_ptr<Flat> flat = std::make_shared<Flat>();
  flat->initialise(5,0.,0.5);
  ModuleList::addModule(flat);

  std::shared_ptr<BoundaryConditionContainer> bcs = std::make_shared<BoundaryConditionContainer>();
  bcs->initialise(std::make_shared<Transmissive>(), std::make_shared<Transmissive>());

  ImplicitSolver::Vector states(10);

  states<<  1.00000000e-06,3.00000000e-01,
            2.25208555e-05,1.52495383e-01,
            3.07299124e-05,3.76788153e-01,
            9.07299124e-05,3.76788153e-01,
            5.62725541e-05,5.17095798e-01;

  std::shared_ptr<ImplicitSolver> solver = std::make_shared<ImplicitSolver>();
  solver->initialise(1.0);
  solver->setBoundaryConditions(bcs);

  ImplicitSolver::SpMatRowMaj J(10,10);

  solver->jacobian(states,0.2,0.,J);

  BOOST_CHECK_CLOSE(J.coeff(0,0),  1.00000000e+00, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(0,1),  0.00000000e+00, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(0,2),  1.52495383e-01, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(0,3),  2.25208555e-05, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(1,0), -2.9603969628985005e+17, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(1,1),  493400001294.12927    , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(1,2), -10515439.669656562    , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(1,3),  0.6745594988564938    , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(1,4),  2113731.6314322767    , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(1,5), -1205.5352671639109    , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(2,0), -3.00000000e-01, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(2,1), -1.00000000e-06, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(2,2),  1.00000000e+00, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(2,3),  0.00000000e+00, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(2,4),  3.76788153e-01, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(2,5),  3.07299124e-05, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(3,0), -77148.556796549878 , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(3,1), -0.18565065267704534, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(3,2), -13173811439414.32  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(3,3),  972813056.99226773 , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(3,4), -267128.37381937256 , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(3,5),  0.2601484667518365 , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(3,6),  15694.890843000663 , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(3,7), -26.445301669030354 , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(4,2), -1.52495383e-01, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(4,3), -2.25208555e-05, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(4,4),  1.00000000e+00, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(4,5),  0.00000000e+00, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(4,6),  3.76788153e-01, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(4,7),  9.07299124e-05, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(5,0),  908488.78373531031  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(5,1), -21.176664669820742  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(5,2), -13965.484679269044  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(5,3), -0.60460461306418289 , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(5,4), -12812429872610.865  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(5,5),  522488324.03927457  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(5,6), -239019.80109609151  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(5,7),  0.69355398082089326 , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(5,8),  120590.41759824802  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(5,9), -91.777859074411268  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(6,4), -3.76788153e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(6,5), -3.07299124e-05, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(6,6),  1.00000000e+00, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(6,7),  0.00000000e+00, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(6,8),  5.17095798e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(6,9),  5.62725541e-05, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(7,2),  4955.9473562172316  , 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(7,3), -5.1080261112411325  , 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(7,4),  40979.575123459777  , 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(7,5), -0.5603417762243208  , 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(7,6), -497786056649.57709  , 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(7,7),  59937451.753089376  , 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(7,8), -49862.143935200475  , 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(7,9),  0.59474723273587971 , 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(8,6), -3.76788153e-01, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(8,7), -9.07299124e-05, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(8,8),  1.00000000e+00, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(8,9),  0.00000000e+00, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(9,4),  78820.030807985604 , 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(9,5), -44.947988649888202 , 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(9,6),  53610.498886193243 , 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(9,7), -0.89675665983509212, 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(9,8), -2863490632061.2012 , 1e-5);
  BOOST_CHECK_CLOSE(J.coeff(9,9),  155813852.06131446 , 1e-5);

  ModuleList::clear();
}

BOOST_AUTO_TEST_CASE(setupJacobianRandPeriodicLeft)
{
  std::shared_ptr<System> sys = std::make_shared<System>(); 
  sys->initialise(1.,1.1,1.2,111,0.12,TimeSpaceDependReal("1.23"),TimeSpaceDependReal("4."));
  ModuleList::addModule(sys);

  std::shared_ptr<Flat> flat = std::make_shared<Flat>();
  flat->initialise(5,0.,0.5);
  ModuleList::addModule(flat);

  std::shared_ptr<BoundaryConditionContainer> bcs = std::make_shared<BoundaryConditionContainer>();
  bcs->initialise(std::make_shared<Periodic>(), std::make_shared<Periodic>());

  ImplicitSolver::Vector states(10);

  states     <<  3.07299124e-05,3.76788153e-01,
                 9.07299124e-05,3.76788153e-01,
                 5.62725541e-05,5.17095798e-01,
                 1.00000000e-06,3.00000000e-01,
                 2.25208555e-05,1.52495383e-01;

  std::shared_ptr<ImplicitSolver> solver = std::make_shared<ImplicitSolver>();
  solver->initialise(1.0);
  solver->setBoundaryConditions(bcs);

  ImplicitSolver::SpMatRowMaj J(10,10);

  solver->jacobian(states,0.2,0.,J);

  BOOST_CHECK_CLOSE(J.coeff(0,8), -1.52495383e-01, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(0,9), -2.25208555e-05, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(0,0),  1.00000000e+00, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(0,1),  0.00000000e+00, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(0,2),  3.76788153e-01, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(0,3),  9.07299124e-05, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(1,6),  908488.78373531031  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(1,7), -21.176664669820742  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(1,8), -13965.484679269044  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(1,9), -0.60460461306418289 , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(1,0), -12812429872610.865  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(1,1),  522488324.03927457  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(1,2), -239019.80109609151  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(1,3),  0.69355398082089326 , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(1,4),  120590.41759824802  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(1,5), -91.777859074411268  , 1e-10);
  ModuleList::clear();
}

BOOST_AUTO_TEST_CASE(setupJacobianRandPeriodicRight)
{
  std::shared_ptr<System> sys = std::make_shared<System>(); 
  sys->initialise(1.,1.1,1.2,111,0.12,TimeSpaceDependReal("1.23"),TimeSpaceDependReal("4."));
  ModuleList::addModule(sys);

  std::shared_ptr<Flat> flat = std::make_shared<Flat>();
  flat->initialise(5,0.,0.5);
  ModuleList::addModule(flat);

  std::shared_ptr<BoundaryConditionContainer> bcs = std::make_shared<BoundaryConditionContainer>();
  bcs->initialise(std::make_shared<Periodic>(), std::make_shared<Periodic>());

  ImplicitSolver::Vector states(10);
  states     <<  9.07299124e-05,3.76788153e-01,
                 5.62725541e-05,5.17095798e-01,
                 1.00000000e-06,3.00000000e-01,
                 2.25208555e-05,1.52495383e-01,
                 3.07299124e-05,3.76788153e-01;
  std::shared_ptr<ImplicitSolver> solver = std::make_shared<ImplicitSolver>();
  solver->initialise(1.0);
  solver->setBoundaryConditions(bcs);

  ImplicitSolver::SpMatRowMaj J(10,10);

  solver->jacobian(states,0.2,0.,J);

  BOOST_CHECK_CLOSE(J.coeff(8,6), -1.52495383e-01, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(8,7), -2.25208555e-05, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(8,8),  1.00000000e+00, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(8,9),  0.00000000e+00, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(8,0),  3.76788153e-01, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(8,1),  9.07299124e-05, 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(9,4),  908488.78373531031  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(9,5), -21.176664669820742  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(9,6), -13965.484679269044  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(9,7), -0.60460461306418289 , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(9,8), -12812429872610.865  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(9,9),  522488324.03927457  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(9,0), -239019.80109609151  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(9,1),  0.69355398082089326 , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(9,2),  120590.41759824802  , 1e-10);
  BOOST_CHECK_CLOSE(J.coeff(9,3), -91.777859074411268  , 1e-10);

  ModuleList::clear();
}

BOOST_AUTO_TEST_CASE(checkValidTest)
{
  std::shared_ptr<System> sys = std::make_shared<System>(); 
  sys->initialise(1.,1.1,1.2,111,0.12,TimeSpaceDependReal("1.23"),TimeSpaceDependReal("4."));
  ModuleList::addModule(sys);

  std::shared_ptr<Flat> flat = std::make_shared<Flat>();
  flat->initialise(5,0.,0.5);
  ModuleList::addModule(flat);

  ImplicitSolver::Vector states(10);
  states     <<  7.55208555e-05,4.52495383e-01,
                 1.00000000e-06,0.00000000e+00,
                -2.73342916e-05,1.36632144e-01,
                 5.62725541e-05,5.17095798e-01,
                 9.07299124e-05,3.76788153e-01;


  std::shared_ptr<ImplicitSolver> solver = std::make_shared<ImplicitSolver>();
  solver->initialise(1.0);

  BOOST_CHECK(solver->checkValid(states));

  BOOST_CHECK_CLOSE(states[0], 7.55208555e-05, 1e-10);
  BOOST_CHECK_CLOSE(states[1], 4.52495383e-01, 1e-10);
  BOOST_CHECK_CLOSE(states[2], 1.00000000e-06, 1e-10);
  BOOST_CHECK_CLOSE(states[3], 0.00000000e+00, 1e-10);
  BOOST_CHECK(states[4] > 0. ); // Negative height should have been corrected
  BOOST_CHECK_CLOSE(states[5], 1.36632144e-01, 1e-10);
  BOOST_CHECK_CLOSE(states[6], 5.62725541e-05, 1e-10);
  BOOST_CHECK_CLOSE(states[7], 5.17095798e-01, 1e-10);
  BOOST_CHECK_CLOSE(states[8], 9.07299124e-05, 1e-10);
  BOOST_CHECK_CLOSE(states[9], 3.76788153e-01, 1e-10);

  ModuleList::clear();
}

BOOST_AUTO_TEST_SUITE_END()

#endif
