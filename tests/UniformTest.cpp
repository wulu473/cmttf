

#include <boost/test/unit_test.hpp>

#include "Uniform.hpp"
#include "Attributes.hpp"

#include "System.hpp"
#ifndef ROBERTS
#warning Uniform unit tests only work for system roberts
#warning Skipping UniformTest.cpp ...
#else

BOOST_AUTO_TEST_SUITE(UniformTests)

BOOST_AUTO_TEST_CASE(setData)
{
  State state;
  state << 1, 2;

  Uniform ic;
  ic.initialise(state);

  std::shared_ptr<DataPatch> data = std::make_shared<DataPatch>(5);

  ic.setData(data);

  BOOST_CHECK_CLOSE((*data)(0,0),1,1e-5);
  BOOST_CHECK_CLOSE((*data)(0,1),2,1e-5);
  BOOST_CHECK_CLOSE((*data)(2,0),1,1e-5);
  BOOST_CHECK_CLOSE((*data)(2,1),2,1e-5);
  BOOST_CHECK_CLOSE((*data)(4,0),1,1e-5);
  BOOST_CHECK_CLOSE((*data)(4,1),2,1e-5);
}

BOOST_AUTO_TEST_SUITE_END()
#endif
