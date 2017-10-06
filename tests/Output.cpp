
#include <boost/test/unit_test.hpp>

#include "Gnuplot.hpp"

BOOST_AUTO_TEST_SUITE(OutputTests)

BOOST_AUTO_TEST_CASE(needsOutput)
{
  const real interval = 0.1;
  const int frequency = 100;

  Gnuplot output;
  output.initialise("test",interval,frequency);

  BOOST_CHECK_EQUAL(output.needsOutput(0.1,0.01,10),true);

  // Accept small errors due to floating point inaccuracies
  BOOST_CHECK_EQUAL(output.needsOutput(0.1+1e-7,0.01,10),true);

  // Error in dt is too large
  BOOST_CHECK_EQUAL(output.needsOutput(0.1+1e-3,0.01,10),false);
}

BOOST_AUTO_TEST_CASE(maxDt)
{
  const real interval = 1.05;
  const int frequency = 100;

  Gnuplot output;
  output.initialise("test",interval,frequency);

  BOOST_CHECK_CLOSE(output.maxDt(1,0.1,10),0.05,1e-5);
}

BOOST_AUTO_TEST_SUITE_END()


