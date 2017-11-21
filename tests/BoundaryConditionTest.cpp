
#include <boost/test/unit_test.hpp>

#include "BoundaryCondition.hpp"

BOOST_AUTO_TEST_SUITE(BoundaryConditionTests)

BOOST_AUTO_TEST_CASE(initBoundaryConditionContainer)
{
  Parameters params;
  params.readFile("tests/ParametersTest.cfg");

  BoundaryConditionContainer cont;
  cont.initialiseFromParameters(params);
}

BOOST_AUTO_TEST_SUITE_END()

