
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

void Flat::initialiseFromFile()
{
  this->initialise(
      //! Number of cells
      getParameter<int>("cells"),

      //! Left edge of domain
      getParameter<real>("min"),

      //! Right edge of domain
      getParameter<real>("max")
  );
}

Coord Flat::x(const real s) const
{
  return Coord(s,0.);
}

