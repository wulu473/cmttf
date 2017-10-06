
#include <boost/test/unit_test.hpp>

#include "Modules.hpp"
#include "Flat.hpp"
#include "Output.hpp"
#include "InitialCondition.hpp"

BOOST_AUTO_TEST_SUITE(ModulesTests)

BOOST_AUTO_TEST_CASE(setAndGetModule)
{
  std::shared_ptr<Flat> chart = std::make_shared<Flat>();
  
  Modules::addModule(chart);

  std::shared_ptr<const Flat> chart2 = Modules::module<Flat>();

  BOOST_CHECK_EQUAL(chart2,chart);

  Modules::clear();
}

BOOST_AUTO_TEST_CASE(initialiseFromFile)
{
  Parameters::readFile("tests/ParametersTest.cfg");
  Modules::initialiseFromFile();

  std::shared_ptr<const Output> output = Modules::module<Output>();
  BOOST_CHECK_EQUAL(output->name(),"Output.Gnuplot");

  std::shared_ptr<const InitialCondition> initialCond = Modules::module<InitialCondition>();
  BOOST_CHECK_EQUAL(initialCond->name(),"InitialCondition.Uniform");

  std::shared_ptr<const Domain> domain = Modules::module<Domain>();
  BOOST_CHECK_EQUAL(domain->name(),"Domain.Flat");

  Modules::clear();
}

BOOST_AUTO_TEST_SUITE_END()
