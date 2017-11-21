
#include "Flat.hpp"

#include "System.hpp"

REGISTERIMPL(Flat);

Flat::Flat()
{

}

Flat::~Flat()
{

}

std::string Flat::moduleName() const
{
  return "Flat";
}

Coord Flat::x(const real s) const
{
  return Coord(s,0.);
}

Coord Flat::x_(const unsigned int i, const real /*s*/) const
{
  Coord basis(0.,0.);
  basis[i] = 1.;
  return basis;
}

real Flat::kappa(const real /*s*/) const
{
  return 0.;
}

real Flat::dkappa_ds(const real /*s*/) const
{
  return 0.;
}

/**
 * N [in] Number of cells
 * L [in] Left edge of domain
 * R [in] Right edge of domain
 *
 */
void Flat::initialise(const int N, const real L, const real R)
{
  m_minS = L;
  m_maxS = R;
  m_nCells = N;
}

void Flat::initialiseFromParameters(const Parameters& params)
{
  this->initialise(
      //! Number of cells
      getParameter<int>(params, "cells"),

      //! Left edge of domain
      getParameter<real>(params, "min"),

      //! Right edge of domain
      getParameter<real>(params, "max")
  );
}


