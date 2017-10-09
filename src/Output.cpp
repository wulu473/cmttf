
#include "Output.hpp"

std::string Output::baseName() const
{
  return "Output";
}

Output::Output()
{

}

Output::~Output()
{

}

bool Output::needsOutput(const real t, const real dt, const unsigned int iter) const
{
  if(iter % m_frequency==0)
  {
    return true;
  }

  // Use dt as a measure for the relative magnitude
  if(fabs(t-m_tLast-m_interval)/dt < 1e-4)
  {
    return true;
  }

  return false;

}

real Output::maxDt(const real t, const real dt, const unsigned int iter) const
{
  return m_tLast+m_interval-t;
}


