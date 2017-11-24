

#include <boost/test/unit_test.hpp>

#include "Cylinder.hpp"

BOOST_AUTO_TEST_SUITE(CylinderTests)

BOOST_AUTO_TEST_CASE(xTest)
{
  Cylinder cyl;
  cyl.initialise(100,0,2*M_PI,2.0);

  Coord x = cyl.x(2*M_PI/2.);

  BOOST_CHECK_SMALL(x[0]   ,1e-10); // Boost uses Knuth's tolerance which has a singularity at 0. So don't use BOOST_CHECK_CLOSE here
  BOOST_CHECK_CLOSE(x[1],2.,1e-10);
}

/**
 * Domain start at the east (default)
 *
 * check if half way is west
 */
BOOST_AUTO_TEST_CASE(sTest)
{
  Cylinder cyl;
  cyl.initialise(101,0,2*M_PI,2.0);

  const real s = cyl.s(50);

  BOOST_CHECK_CLOSE(s,2*M_PI,1e-10);
}

/**
 * Domain starts at the north
 *
 * check if half way point is south
 *
 */
BOOST_AUTO_TEST_CASE(nonZeroOriginTest)
{
  Cylinder cyl;
  cyl.initialise(101,M_PI/2.,5*M_PI/2.,3.0);

  const real s = cyl.s(50);
  const Coord x = cyl.x(s);

  BOOST_CHECK_SMALL(x[0],    1e-10);
  BOOST_CHECK_CLOSE(x[1],-3.,1e-10);
}

BOOST_AUTO_TEST_SUITE_END()
