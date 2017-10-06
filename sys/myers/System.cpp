
#include <iomanip>

#include "System.hpp"

State System::F(const StencilArray& states, const real dx, const real x, const real t) const
{
  const real beta = m_beta(t,x);
  const real sigma = m_sigma;
  const real mu = m_mu;
  const real tau = m_tau(t,x);

  const real ST_0_0 = states[0][0];
  const real ST_1_0 = states[1][0];
  const real ST_2_0 = states[2][0];
  const real ST_3_0 = states[3][0];
  const real ST_4_0 = states[4][0];

  State F;
  F[0] = -1.0L/24.0L*ST_0_0*pow(ST_1_0, 3)*sigma/(pow(dx, 4)*mu) - 1.0L/8.0L*ST_0_0*pow(ST_1_0, 2)*ST_2_0*sigma/(pow(dx, 4)*mu) - 1.0L/8.0L*ST_0_0*ST_1_0*pow(ST_2_0, 2)*sigma/(pow(dx, 4)*mu) - 1.0L/24.0L*ST_0_0*pow(ST_2_0, 3)*sigma/(pow(dx, 4)*mu) + (1.0L/8.0L)*pow(ST_1_0, 4)*sigma/(pow(dx, 4)*mu) + (1.0L/4.0L)*pow(ST_1_0, 3)*ST_2_0*sigma/(pow(dx, 4)*mu) + (1.0L/24.0L)*pow(ST_1_0, 3)*ST_3_0*sigma/(pow(dx, 4)*mu) + (1.0L/8.0L)*pow(ST_1_0, 2)*ST_2_0*ST_3_0*sigma/(pow(dx, 4)*mu) - 1.0L/8.0L*pow(ST_1_0, 2)*tau/(dx*mu) - 5.0L/24.0L*ST_1_0*pow(ST_2_0, 3)*sigma/(pow(dx, 4)*mu) + (1.0L/4.0L)*ST_1_0*pow(ST_2_0, 2)*ST_3_0*sigma/(pow(dx, 4)*mu) + (1.0L/8.0L)*ST_1_0*ST_2_0*pow(ST_3_0, 2)*sigma/(pow(dx, 4)*mu) - 1.0L/4.0L*ST_1_0*ST_2_0*tau/(dx*mu) + (1.0L/24.0L)*ST_1_0*pow(ST_3_0, 3)*sigma/(pow(dx, 4)*mu) - 1.0L/4.0L*pow(ST_2_0, 4)*sigma/(pow(dx, 4)*mu) - 5.0L/24.0L*pow(ST_2_0, 3)*ST_3_0*sigma/(pow(dx, 4)*mu) - 1.0L/24.0L*pow(ST_2_0, 3)*ST_4_0*sigma/(pow(dx, 4)*mu) - 1.0L/8.0L*pow(ST_2_0, 2)*ST_3_0*ST_4_0*sigma/(pow(dx, 4)*mu) + (1.0L/4.0L)*ST_2_0*pow(ST_3_0, 3)*sigma/(pow(dx, 4)*mu) - 1.0L/8.0L*ST_2_0*pow(ST_3_0, 2)*ST_4_0*sigma/(pow(dx, 4)*mu) + (1.0L/4.0L)*ST_2_0*ST_3_0*tau/(dx*mu) + (1.0L/8.0L)*pow(ST_3_0, 4)*sigma/(pow(dx, 4)*mu) - 1.0L/24.0L*pow(ST_3_0, 3)*ST_4_0*sigma/(pow(dx, 4)*mu) + (1.0L/8.0L)*pow(ST_3_0, 2)*tau/(dx*mu) - beta;

  return F;
}

StencilJacobian System::J(const StencilArray& states,
            const real dx, const real x, const real t) const
{
  const real sigma = m_sigma;
  const real mu = m_mu;
  const real tau = m_tau(t,x);

  const real ST_0_0 = states[0][0];
  const real ST_1_0 = states[1][0];
  const real ST_2_0 = states[2][0];
  const real ST_3_0 = states[3][0];
  const real ST_4_0 = states[4][0];

  StencilJacobian J;
  J(0,0) = -1.0L/24.0L*pow(ST_1_0, 3)*sigma/(pow(dx, 4)*mu) - 1.0L/8.0L*pow(ST_1_0, 2)*ST_2_0*sigma/(pow(dx, 4)*mu) - 1.0L/8.0L*ST_1_0*pow(ST_2_0, 2)*sigma/(pow(dx, 4)*mu) - 1.0L/24.0L*pow(ST_2_0, 3)*sigma/(pow(dx, 4)*mu);
  J(0,1) = -1.0L/8.0L*ST_0_0*pow(ST_1_0, 2)*sigma/(pow(dx, 4)*mu) - 1.0L/4.0L*ST_0_0*ST_1_0*ST_2_0*sigma/(pow(dx, 4)*mu) - 1.0L/8.0L*ST_0_0*pow(ST_2_0, 2)*sigma/(pow(dx, 4)*mu) + (1.0L/2.0L)*pow(ST_1_0, 3)*sigma/(pow(dx, 4)*mu) + (3.0L/4.0L)*pow(ST_1_0, 2)*ST_2_0*sigma/(pow(dx, 4)*mu) + (1.0L/8.0L)*pow(ST_1_0, 2)*ST_3_0*sigma/(pow(dx, 4)*mu) + (1.0L/4.0L)*ST_1_0*ST_2_0*ST_3_0*sigma/(pow(dx, 4)*mu) - 1.0L/4.0L*ST_1_0*tau/(dx*mu) - 5.0L/24.0L*pow(ST_2_0, 3)*sigma/(pow(dx, 4)*mu) + (1.0L/4.0L)*pow(ST_2_0, 2)*ST_3_0*sigma/(pow(dx, 4)*mu) + (1.0L/8.0L)*ST_2_0*pow(ST_3_0, 2)*sigma/(pow(dx, 4)*mu) - 1.0L/4.0L*ST_2_0*tau/(dx*mu) + (1.0L/24.0L)*pow(ST_3_0, 3)*sigma/(pow(dx, 4)*mu);
  J(0,2) = -1.0L/8.0L*ST_0_0*pow(ST_1_0, 2)*sigma/(pow(dx, 4)*mu) - 1.0L/4.0L*ST_0_0*ST_1_0*ST_2_0*sigma/(pow(dx, 4)*mu) - 1.0L/8.0L*ST_0_0*pow(ST_2_0, 2)*sigma/(pow(dx, 4)*mu) + (1.0L/4.0L)*pow(ST_1_0, 3)*sigma/(pow(dx, 4)*mu) + (1.0L/8.0L)*pow(ST_1_0, 2)*ST_3_0*sigma/(pow(dx, 4)*mu) - 5.0L/8.0L*ST_1_0*pow(ST_2_0, 2)*sigma/(pow(dx, 4)*mu) + (1.0L/2.0L)*ST_1_0*ST_2_0*ST_3_0*sigma/(pow(dx, 4)*mu) + (1.0L/8.0L)*ST_1_0*pow(ST_3_0, 2)*sigma/(pow(dx, 4)*mu) - 1.0L/4.0L*ST_1_0*tau/(dx*mu) - pow(ST_2_0, 3)*sigma/(pow(dx, 4)*mu) - 5.0L/8.0L*pow(ST_2_0, 2)*ST_3_0*sigma/(pow(dx, 4)*mu) - 1.0L/8.0L*pow(ST_2_0, 2)*ST_4_0*sigma/(pow(dx, 4)*mu) - 1.0L/4.0L*ST_2_0*ST_3_0*ST_4_0*sigma/(pow(dx, 4)*mu) + (1.0L/4.0L)*pow(ST_3_0, 3)*sigma/(pow(dx, 4)*mu) - 1.0L/8.0L*pow(ST_3_0, 2)*ST_4_0*sigma/(pow(dx, 4)*mu) + (1.0L/4.0L)*ST_3_0*tau/(dx*mu);
  J(0,3) = (1.0L/24.0L)*pow(ST_1_0, 3)*sigma/(pow(dx, 4)*mu) + (1.0L/8.0L)*pow(ST_1_0, 2)*ST_2_0*sigma/(pow(dx, 4)*mu) + (1.0L/4.0L)*ST_1_0*pow(ST_2_0, 2)*sigma/(pow(dx, 4)*mu) + (1.0L/4.0L)*ST_1_0*ST_2_0*ST_3_0*sigma/(pow(dx, 4)*mu) + (1.0L/8.0L)*ST_1_0*pow(ST_3_0, 2)*sigma/(pow(dx, 4)*mu) - 5.0L/24.0L*pow(ST_2_0, 3)*sigma/(pow(dx, 4)*mu) - 1.0L/8.0L*pow(ST_2_0, 2)*ST_4_0*sigma/(pow(dx, 4)*mu) + (3.0L/4.0L)*ST_2_0*pow(ST_3_0, 2)*sigma/(pow(dx, 4)*mu) - 1.0L/4.0L*ST_2_0*ST_3_0*ST_4_0*sigma/(pow(dx, 4)*mu) + (1.0L/4.0L)*ST_2_0*tau/(dx*mu) + (1.0L/2.0L)*pow(ST_3_0, 3)*sigma/(pow(dx, 4)*mu) - 1.0L/8.0L*pow(ST_3_0, 2)*ST_4_0*sigma/(pow(dx, 4)*mu) + (1.0L/4.0L)*ST_3_0*tau/(dx*mu);
  J(0,4) = -1.0L/24.0L*pow(ST_2_0, 3)*sigma/(pow(dx, 4)*mu) - 1.0L/8.0L*pow(ST_2_0, 2)*ST_3_0*sigma/(pow(dx, 4)*mu) - 1.0L/8.0L*ST_2_0*pow(ST_3_0, 2)*sigma/(pow(dx, 4)*mu) - 1.0L/24.0L*pow(ST_3_0, 3)*sigma/(pow(dx, 4)*mu);

  return J;
}


void System::initialise(const real mu, const real sigma,
          const TimeSpaceDependReal tau, const TimeSpaceDependReal beta)
{
  m_mu = mu;
  m_tau = tau;
  m_sigma = sigma;
  m_beta = beta;
}

void System::initialiseFromFile()
{
  const real mu = getParameter<real>("mu");
  const TimeSpaceDependReal tau = getParameter<TimeSpaceDependReal>("tau");
  const TimeSpaceDependReal beta = getParameter<TimeSpaceDependReal>("beta");
  const real sigma = getParameter<real>("sigma");

  this->initialise(mu,sigma,tau,beta);
}
