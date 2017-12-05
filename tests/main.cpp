#define BOOST_TEST_MODULE Main

#include "Log.hpp"

#include "Attributes.hpp"

#include <boost/test/unit_test.hpp>

struct LogSetup
{
  LogSetup()
  {
    // This is not a very portable way of setting the log level. 
    // It doesn't compile on many boost versions
    // TODO Implement a more robust way of setting the log level
    /*
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
    */
  }
  ~LogSetup() { }
};

BOOST_GLOBAL_FIXTURE( LogSetup) ;
