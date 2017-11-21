
#include <boost/test/unit_test.hpp>

#include "ModuleList.hpp"
#include "Flat.hpp"
#include "Output.hpp"
#include "InitialCondition.hpp"

#include "System.hpp"
#ifndef ROBERTS
#warning All of ModuleList unit tests only work for system roberts
#endif

BOOST_AUTO_TEST_SUITE(ModuleListTests)

BOOST_AUTO_TEST_CASE(setAndGetModule)
{
  std::shared_ptr<Flat> chart = std::make_shared<Flat>();
  
  ModuleList::addModule(chart);

  std::shared_ptr<const Flat> chart2 = ModuleList::module<Flat>();

  BOOST_CHECK_EQUAL(chart2,chart);

  ModuleList::clear();
}

#ifndef ROBERTS
#warning Skipping initialiseFromFile test...
#else
BOOST_AUTO_TEST_CASE(initialiseFromFile)
{
  Parameters params;
  params.readFile("tests/ParametersTest.cfg");
  ModuleList::initialiseFromParameters(params);

  std::shared_ptr<const Output> output = ModuleList::module<Output>();
  BOOST_CHECK_EQUAL(output->name(),"Output.Gnuplot");

  std::shared_ptr<const InitialCondition> initialCond = ModuleList::module<InitialCondition>();
  BOOST_CHECK_EQUAL(initialCond->name(),"InitialCondition.Uniform");

  std::shared_ptr<const Domain> domain = ModuleList::module<Domain>();
  BOOST_CHECK_EQUAL(domain->name(),"Domain.Flat");

  ModuleList::clear();
}
#endif

BOOST_AUTO_TEST_SUITE_END()
