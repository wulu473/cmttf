
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>

#include "NewtonTrustRegion.hpp"


BOOST_AUTO_TEST_SUITE(NewtonTrustRegionTests)

void f(const RootFinder::EVector& x, RootFinder::EVector& f)
{
  f[0] = 3*x[0] - cos(x[1]*x[2]) - 3./2.;
  f[1] = 4*x[0]*x[0] - 625*x[1]*x[1] + 2*x[2] - 1;
  f[2] = 20*x[2] + exp(-x[0]*x[1]) + 9;
}

void J(const RootFinder::EVector& x, RootFinder::ESpMatRowMaj& J)
{
  // In this case J isn't sparse but it's just for testing purposes
  typedef Eigen::Triplet<real> T;
  std::vector<T> coeffs;
  coeffs.push_back(T(0,0,3.));
  coeffs.push_back(T(1,0,8*x[0]));
  coeffs.push_back(T(2,0,-x[1]*exp(-x[0]*x[1])));

  coeffs.push_back(T(0,1,x[2]*sin(x[1]*x[2])));
  coeffs.push_back(T(1,1,-1250*x[1]));
  coeffs.push_back(T(2,1,2.));

  coeffs.push_back(T(0,2,x[1]*sin(x[1]*x[2])));
  coeffs.push_back(T(1,2,2.));
  coeffs.push_back(T(2,2,20.));
  J.setFromTriplets(coeffs.begin(),coeffs.end());
}



BOOST_AUTO_TEST_CASE(test1)
{
  std::function<void(const RootFinder::EVector&, RootFinder::EVector&)> fun = f;
  std::function<void(const RootFinder::EVector&, RootFinder::ESpMatRowMaj&)> jac = J;
  RootFinder::EVector x(3);

  // Guess solution
  x << 1., 1., 1.;

  NewtonTrustRegion *method = new NewtonTrustRegion();

  method->solveSparse(fun,jac,x);

  BOOST_CHECK_CLOSE(x[0], 0.8332816138167559,1e-9);
  BOOST_CHECK_CLOSE(x[1], 0.035334616139489156,1e-9);
  BOOST_CHECK_CLOSE(x[2],-0.49854927781103725,1e-9);

  delete method;
}

BOOST_AUTO_TEST_SUITE_END()
