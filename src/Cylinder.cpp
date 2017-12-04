
#include "Cylinder.hpp"

#include "System.hpp"

REGISTERIMPL(Cylinder);

Cylinder::Cylinder()
{

}

Cylinder::~Cylinder()
{

}

std::string Cylinder::moduleName() const
{
  return "Cylinder";
}

Coord Cylinder::x(const real s) const
{
  const real theta = s/m_radius;
  return m_radius*Coord(cos(theta),
                        sin(theta));
}

Coord Cylinder::n(const real s) const
{
  const real theta = s/m_radius;
  return Coord(cos(theta),sin(theta));
}

Coord Cylinder::x_(const unsigned int i, const real s) const
{
#ifdef DEBUG
  assert(i==0 || i==1);
#endif

  const real theta = s/m_radius;
  if(i==0)
  {
    return Coord(-sin(theta),
                  cos(theta));
  }
  else
  {
    return Coord(cos(theta),
                 sin(theta));
  }
}

real Cylinder::kappa(const real /*s*/) const
{
  return 1./m_radius;
}

real Cylinder::dkappa_ds(const real /*s*/) const
{
  return 0.;
}

/**
 * N [in] Number of cells
 * thetaMin [in] Angle in rad of -left- edge of domain
 * thetaMax [in] Angle in rad of -right- edge of domain
 * radius [in] Length of radius
 *
 */
void Cylinder::initialise(const int N, const real thetaMin, const real thetaMax, const real radius)
{
  m_radius = radius;
  m_minS = thetaMin*radius;
  m_maxS = thetaMax*radius;

  assert(radius > std::numeric_limits<real>::epsilon());
  assert(m_minS < m_maxS);
  assert(thetaMax - thetaMin <= 2*M_PI + 10*std::numeric_limits<real>::epsilon());

  m_nCells = N;
}

void Cylinder::initialiseFromParameters(const Parameters& params)
{
  this->initialise(
      //! Number of cells
      getParameter<int>(params, "cells"),

      //! Left edge of domain (rad)
      getParameter<real>(params, "min"),

      //! Right edge of domain (rad)
      getParameter<real>(params, "max"),

      //! Radius of cylinder
      getParameter<real>(params, "rad")
  );
}


