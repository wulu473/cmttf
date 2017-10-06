
#include <iostream>

#include "Log.hpp"


Log::Level Log::stringToLevel(const std::string levelStr)
{
  const std::map<std::string,Log::Level> stringToLevel = {
       {"debug",Level::debug}
      ,{"info",Level::info}
      ,{"warning",Level::warning}
      ,{"error",Level::error} 
      ,{"fatal",Level::fatal} 
    };

  try
  { 
    Level level = stringToLevel.at(levelStr);
    return level;
  }
  catch( const std::out_of_range& oor)
  {
    std::cerr << "Cannot convert '" << levelStr << "' to a valid log level. "
              << "Should be one of:" << std::endl;
    for(auto level : stringToLevel)
    {
      std::cerr << "   - '" << level.first << "'" << std::endl;
    }
    exit(2);
  }
  return Log::Level::info;
}

void Log::setLevel(const Log::Level level)
{
  boost::log::core::get()->set_filter
    (
     boost::log::trivial::severity >= level
    );
}

