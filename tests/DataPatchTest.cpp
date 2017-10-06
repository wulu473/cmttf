#include <boost/test/unit_test.hpp>

#include "DataPatch.hpp"

BOOST_AUTO_TEST_SUITE(DataPatchTest)

BOOST_AUTO_TEST_CASE(constructors)
{
  std::shared_ptr<DataPatch> data = std::make_shared<DataPatch>(5);

  BOOST_CHECK_EQUAL(data->size(),5*SystemAttributes::stateSize);
  BOOST_CHECK_EQUAL(data->cols(),SystemAttributes::stateSize);
  BOOST_CHECK_EQUAL(data->rows(),5);
}

BOOST_AUTO_TEST_SUITE_END()
