#define BOOST_TEST_MODULE Main

#include "Log.hpp"

#include "Attributes.hpp"

#include <boost/test/unit_test.hpp>

struct LogSetup
{
  LogSetup() { Log::setLevel(Log::Level::error);}
  ~LogSetup() { }
};

BOOST_GLOBAL_FIXTURE( LogSetup) ;
