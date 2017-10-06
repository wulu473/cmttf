
#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <iomanip>

namespace Log
{
  typedef boost::log::trivial::severity_level Level;

  void setLevel(const Level);
 
  Level stringToLevel(const std::string);
};


