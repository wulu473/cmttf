
#include "Flat.hpp"

#include "Modules.hpp"
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

/**
 * N [in] Number of cells
 * L [in] Left edge of domain
 * R [in] Right edge of domain
 *
 */
void Flat::initialise(const int N, const real L, const real R)
{
  m_xL = L;
  m_xR = R;
  m_iL = 0;
  m_iR = N;
  m_dx = (R-L)/N;
  m_N = N;
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

real Flat::x(const int i) const
{
  return m_xL + (i-m_iL)*m_dx + m_dx/2.;
}

real Flat::dx() const
{
  return m_dx;
}

real Flat::minX() const
{
  return m_xL;
}

real Flat::maxX() const
{
  return m_xR;
}

int Flat::begin() const
{
  return m_iL;
}

int Flat::end() const
{
  return m_iR;
}

unsigned int Flat::cells() const
{
  return m_N;
}

