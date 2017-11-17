
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
Coord Domain::x(const real) const
{
  BOOST_LOG_TRIVIAL(error) << "Domain::x called from base class.";
  throw InheritanceException();
  return Coord(0.);
}

//! Surface coordinate
real Domain::s(const int i) const
{
  return minS() + (i - begin())*ds() + ds()/2.;
}

//! Return surface curvature
real Domain::kappa(const real /*s*/) const
{
  BOOST_LOG_TRIVIAL(error) << "Domain::kappa called from base class.";
  throw InheritanceException();
  return 0.;
}

//! Return derivative of surface curvature
real Domain::dKappa_ds(const real /*s*/) const
{
  BOOST_LOG_TRIVIAL(error) << "Domain::dKappa_ds called from base class.";
  throw InheritanceException();
  return 0.;
}

//! Grid spacing
real Domain::ds() const
{
  return (maxS()-minS())/cells();
}

//! Coordinate of left edge of surface
real Domain::minS() const
{
  return m_minS;
}

//! Coordinate of right edge of surface
real Domain::maxS() const
{
  return m_maxS;
}

//! Index of the left most cell
int Domain::begin() const
{
  return 0;
}

//! Index of the right most cell +1
int Domain::end() const
{
  return m_nCells;
}

//! Number of cells of computational domain
unsigned int Domain::cells() const
{
  return m_nCells;
}


