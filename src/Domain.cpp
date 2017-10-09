
#include <boost/log/trivial.hpp>

#include "Domain.hpp"

Domain::Domain()
{
}

Domain::~Domain()
{
}

std::string Domain::baseName() const
{
  return "Domain";
}

void Domain::initialise()
{
}

//! Coordinate of cell centre
real Domain::x(const int) const
{
  BOOST_LOG_TRIVIAL(error) << "Domain::x called from base class.";
  throw InheritanceException();
  return 0;
}

//! Grid spacing
real Domain::dx() const
{
  BOOST_LOG_TRIVIAL(error) << "Domain::dx called from base class.";
  throw InheritanceException();
  return 0;
}

//! Coordinate of left edge of the domain
real Domain::minX() const
{
  BOOST_LOG_TRIVIAL(error) << "Domain::minX called from base class.";
  throw InheritanceException();
  return 0;

}

//! Coordinate of right edge of the domain
real Domain::maxX() const
{
  BOOST_LOG_TRIVIAL(error) << "Domain::maxX called from base class.";
  throw InheritanceException();
  return 0;
}

//! Index of the left most cell
int Domain::begin() const
{
  BOOST_LOG_TRIVIAL(error) << "Domain::begin called from base class.";
  throw InheritanceException();
  return 0;
}

//! Index of the right most cell +1
int Domain::end() const
{
  BOOST_LOG_TRIVIAL(error) << "Domain::end called from base class.";
  throw InheritanceException();
  return 0;
}

//! Number of cells of computational domain
unsigned int Domain::cells() const
{
  BOOST_LOG_TRIVIAL(error) << "Domain::cells called from base class.";
  throw InheritanceException();
  return 0;
}


