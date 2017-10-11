#define BOOST_TEST_MODULE Main

#include "Log.hpp"

#include "Attributes.hpp"

#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_parameters.hpp>

struct LogSetup
{
  LogSetup()
  {
    if(boost::unit_test::runtime_config::get<boost::unit_test::log_level>( 
          boost::unit_test::runtime_config::LOG_LEVEL 
          ) <= boost::unit_test::log_messages)
    {
      Log::setLevel(Log::Level::debug);
    }
    else
    {
      Log::setLevel(Log::Level::error);
    }
  }
  ~LogSetup() { }
};

BOOST_GLOBAL_FIXTURE( LogSetup) ;
