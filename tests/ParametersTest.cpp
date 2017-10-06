
#include <boost/test/unit_test.hpp>

#include "Parameters.hpp"


BOOST_AUTO_TEST_SUITE(ParametersTest)

BOOST_AUTO_TEST_CASE(allActiveModulesTest)
{
  Parameters::readFile("tests/ParametersTest.cfg");
  auto modList = Parameters::allActiveModules();
  std::vector<std::string> mods{ std::make_move_iterator(std::begin(modList)), 
                                 std::make_move_iterator(std::end(modList)) };
  std::sort(mods.begin(), mods.end());

  BOOST_CHECK_EQUAL(mods.size(),3);
  BOOST_CHECK_EQUAL(mods[0],"Flat");
  BOOST_CHECK_EQUAL(mods[1],"Gnuplot");
  BOOST_CHECK_EQUAL(mods[2],"Uniform");

}

BOOST_AUTO_TEST_CASE(expressions)
{
  Parameters::readFile("tests/ParametersTest.cfg");
 
  std::function<real(real,real)> exp = Parameters::getExpressionParameter("ExpressionTest.expression");


  BOOST_CHECK_CLOSE(exp(0,0),1,1e-5);
  BOOST_CHECK_CLOSE(exp(0,0.25),0.939413,1e-5);

}

BOOST_AUTO_TEST_CASE(copyExpressions)
{
  Parameters::readFile("tests/ParametersTest.cfg");
 
  const std::function<real(real,real)> exp = Parameters::getExpressionParameter("ExpressionTest.expression");

  std::function<real(real,real)> otherExp(exp);

  BOOST_CHECK_CLOSE(otherExp(0,0),1,1e-5);
  BOOST_CHECK_CLOSE(otherExp(0,0.25),0.939413,1e-5);
}

BOOST_AUTO_TEST_SUITE_END()
