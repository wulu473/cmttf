
#include <iomanip>

#include "System.hpp"

State System::F(const StencilArray& states, const real dx, const real x, const real t) const
{
  const real beta = m_beta(t,x);
  const real sigma = m_sigma;
  const real mu = m_mu;
  const real tau = m_tau(t,x);

  const real u_0_0 = states[0][0];
  const real u_1_0 = states[1][0];
  const real u_2_0 = states[2][0];
  const real u_3_0 = states[3][0];
  const real u_4_0 = states[4][0];

  State F;
  F[0] = -beta - 1.0L/8.0L*tau*pow(u_1_0, 2)/(dx*mu) - 1.0L/4.0L*tau*u_1_0*u_2_0/(dx*mu) + (1.0L/4.0L)*tau*u_2_0*u_3_0/(dx*mu) + (1.0L/8.0L)*tau*pow(u_3_0, 2)/(dx*mu) + (1.0L/24.0L)*sigma*u_0_0*pow(u_1_0, 3)/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*u_0_0*pow(u_1_0, 2)*u_2_0/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*u_0_0*u_1_0*pow(u_2_0, 2)/(pow(dx, 4)*mu) + (1.0L/24.0L)*sigma*u_0_0*pow(u_2_0, 3)/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*pow(u_1_0, 4)/(pow(dx, 4)*mu) - 1.0L/4.0L*sigma*pow(u_1_0, 3)*u_2_0/(pow(dx, 4)*mu) - 1.0L/24.0L*sigma*pow(u_1_0, 3)*u_3_0/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*pow(u_1_0, 2)*u_2_0*u_3_0/(pow(dx, 4)*mu) + (5.0L/24.0L)*sigma*u_1_0*pow(u_2_0, 3)/(pow(dx, 4)*mu) - 1.0L/4.0L*sigma*u_1_0*pow(u_2_0, 2)*u_3_0/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*u_1_0*u_2_0*pow(u_3_0, 2)/(pow(dx, 4)*mu) - 1.0L/24.0L*sigma*u_1_0*pow(u_3_0, 3)/(pow(dx, 4)*mu) + (1.0L/4.0L)*sigma*pow(u_2_0, 4)/(pow(dx, 4)*mu) + (5.0L/24.0L)*sigma*pow(u_2_0, 3)*u_3_0/(pow(dx, 4)*mu) + (1.0L/24.0L)*sigma*pow(u_2_0, 3)*u_4_0/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*pow(u_2_0, 2)*u_3_0*u_4_0/(pow(dx, 4)*mu) - 1.0L/4.0L*sigma*u_2_0*pow(u_3_0, 3)/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*u_2_0*pow(u_3_0, 2)*u_4_0/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*pow(u_3_0, 4)/(pow(dx, 4)*mu) + (1.0L/24.0L)*sigma*pow(u_3_0, 3)*u_4_0/(pow(dx, 4)*mu);
  return F;
}

//! Return the function (Non-linear terms are replaced by estimates using the states at the previous time step)
State System::FLinear(const StencilArray& states_new, const StencilArray& states_old, 
            const real dx, const real x, const real t) const
{
  const real beta = m_beta(t,x);
  const real sigma = m_sigma;
  const real mu = m_mu;
  const real tau = m_tau(t,x);

  const real u_0_0 = states_new[0][0];
  const real u_1_0 = states_new[1][0];
  const real u_2_0 = states_new[2][0];
  const real u_3_0 = states_new[3][0];
  const real u_4_0 = states_new[4][0];
  const real u0_0_0 = states_old[0][0];
  const real u0_1_0 = states_old[1][0];
  const real u0_2_0 = states_old[2][0];
  const real u0_3_0 = states_old[3][0];
  const real u0_4_0 = states_old[4][0];

  State F_lin;
  F_lin[0] = -beta - 1.0L/8.0L*tau*pow(u0_1_0, 2)/(dx*mu) - 1.0L/4.0L*tau*u0_1_0*u0_2_0/(dx*mu) + (1.0L/4.0L)*tau*u0_2_0*u0_3_0/(dx*mu) + (1.0L/8.0L)*tau*pow(u0_3_0, 2)/(dx*mu) + (1.0L/24.0L)*sigma*pow(u0_1_0, 3)*u_0_0/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*pow(u0_1_0, 3)*u_1_0/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*pow(u0_1_0, 3)*u_2_0/(pow(dx, 4)*mu) - 1.0L/24.0L*sigma*pow(u0_1_0, 3)*u_3_0/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*pow(u0_1_0, 2)*u0_2_0*u_0_0/(pow(dx, 4)*mu) - 3.0L/8.0L*sigma*pow(u0_1_0, 2)*u0_2_0*u_1_0/(pow(dx, 4)*mu) + (3.0L/8.0L)*sigma*pow(u0_1_0, 2)*u0_2_0*u_2_0/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*pow(u0_1_0, 2)*u0_2_0*u_3_0/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*u0_1_0*pow(u0_2_0, 2)*u_0_0/(pow(dx, 4)*mu) - 3.0L/8.0L*sigma*u0_1_0*pow(u0_2_0, 2)*u_1_0/(pow(dx, 4)*mu) + (3.0L/8.0L)*sigma*u0_1_0*pow(u0_2_0, 2)*u_2_0/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*u0_1_0*pow(u0_2_0, 2)*u_3_0/(pow(dx, 4)*mu) + (1.0L/24.0L)*sigma*pow(u0_2_0, 3)*u_0_0/(pow(dx, 4)*mu) - 1.0L/6.0L*sigma*pow(u0_2_0, 3)*u_1_0/(pow(dx, 4)*mu) + (1.0L/4.0L)*sigma*pow(u0_2_0, 3)*u_2_0/(pow(dx, 4)*mu) - 1.0L/6.0L*sigma*pow(u0_2_0, 3)*u_3_0/(pow(dx, 4)*mu) + (1.0L/24.0L)*sigma*pow(u0_2_0, 3)*u_4_0/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*pow(u0_2_0, 2)*u0_3_0*u_1_0/(pow(dx, 4)*mu) + (3.0L/8.0L)*sigma*pow(u0_2_0, 2)*u0_3_0*u_2_0/(pow(dx, 4)*mu) - 3.0L/8.0L*sigma*pow(u0_2_0, 2)*u0_3_0*u_3_0/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*pow(u0_2_0, 2)*u0_3_0*u_4_0/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*u0_2_0*pow(u0_3_0, 2)*u_1_0/(pow(dx, 4)*mu) + (3.0L/8.0L)*sigma*u0_2_0*pow(u0_3_0, 2)*u_2_0/(pow(dx, 4)*mu) - 3.0L/8.0L*sigma*u0_2_0*pow(u0_3_0, 2)*u_3_0/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*u0_2_0*pow(u0_3_0, 2)*u_4_0/(pow(dx, 4)*mu) - 1.0L/24.0L*sigma*pow(u0_3_0, 3)*u_1_0/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*pow(u0_3_0, 3)*u_2_0/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*pow(u0_3_0, 3)*u_3_0/(pow(dx, 4)*mu) + (1.0L/24.0L)*sigma*pow(u0_3_0, 3)*u_4_0/(pow(dx, 4)*mu);
  return F_lin;
}

StencilJacobian System::J(const StencilArray& states,
            const real dx, const real x, const real t) const
{
  const real sigma = m_sigma;
  const real mu = m_mu;
  const real tau = m_tau(t,x);

  const real u_0_0 = states[0][0];
  const real u_1_0 = states[1][0];
  const real u_2_0 = states[2][0];
  const real u_3_0 = states[3][0];
  const real u_4_0 = states[4][0];

  StencilJacobian J;
  J(0,0) = (1.0L/24.0L)*sigma*pow(u_1_0, 3)/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*pow(u_1_0, 2)*u_2_0/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*u_1_0*pow(u_2_0, 2)/(pow(dx, 4)*mu) + (1.0L/24.0L)*sigma*pow(u_2_0, 3)/(pow(dx, 4)*mu);
  J(0,1) = -1.0L/4.0L*tau*u_1_0/(dx*mu) - 1.0L/4.0L*tau*u_2_0/(dx*mu) + (1.0L/8.0L)*sigma*u_0_0*pow(u_1_0, 2)/(pow(dx, 4)*mu) + (1.0L/4.0L)*sigma*u_0_0*u_1_0*u_2_0/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*u_0_0*pow(u_2_0, 2)/(pow(dx, 4)*mu) - 1.0L/2.0L*sigma*pow(u_1_0, 3)/(pow(dx, 4)*mu) - 3.0L/4.0L*sigma*pow(u_1_0, 2)*u_2_0/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*pow(u_1_0, 2)*u_3_0/(pow(dx, 4)*mu) - 1.0L/4.0L*sigma*u_1_0*u_2_0*u_3_0/(pow(dx, 4)*mu) + (5.0L/24.0L)*sigma*pow(u_2_0, 3)/(pow(dx, 4)*mu) - 1.0L/4.0L*sigma*pow(u_2_0, 2)*u_3_0/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*u_2_0*pow(u_3_0, 2)/(pow(dx, 4)*mu) - 1.0L/24.0L*sigma*pow(u_3_0, 3)/(pow(dx, 4)*mu);
  J(0,2) = -1.0L/4.0L*tau*u_1_0/(dx*mu) + (1.0L/4.0L)*tau*u_3_0/(dx*mu) + (1.0L/8.0L)*sigma*u_0_0*pow(u_1_0, 2)/(pow(dx, 4)*mu) + (1.0L/4.0L)*sigma*u_0_0*u_1_0*u_2_0/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*u_0_0*pow(u_2_0, 2)/(pow(dx, 4)*mu) - 1.0L/4.0L*sigma*pow(u_1_0, 3)/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*pow(u_1_0, 2)*u_3_0/(pow(dx, 4)*mu) + (5.0L/8.0L)*sigma*u_1_0*pow(u_2_0, 2)/(pow(dx, 4)*mu) - 1.0L/2.0L*sigma*u_1_0*u_2_0*u_3_0/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*u_1_0*pow(u_3_0, 2)/(pow(dx, 4)*mu) + sigma*pow(u_2_0, 3)/(pow(dx, 4)*mu) + (5.0L/8.0L)*sigma*pow(u_2_0, 2)*u_3_0/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*pow(u_2_0, 2)*u_4_0/(pow(dx, 4)*mu) + (1.0L/4.0L)*sigma*u_2_0*u_3_0*u_4_0/(pow(dx, 4)*mu) - 1.0L/4.0L*sigma*pow(u_3_0, 3)/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*pow(u_3_0, 2)*u_4_0/(pow(dx, 4)*mu);
  J(0,3) = (1.0L/4.0L)*tau*u_2_0/(dx*mu) + (1.0L/4.0L)*tau*u_3_0/(dx*mu) - 1.0L/24.0L*sigma*pow(u_1_0, 3)/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*pow(u_1_0, 2)*u_2_0/(pow(dx, 4)*mu) - 1.0L/4.0L*sigma*u_1_0*pow(u_2_0, 2)/(pow(dx, 4)*mu) - 1.0L/4.0L*sigma*u_1_0*u_2_0*u_3_0/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*u_1_0*pow(u_3_0, 2)/(pow(dx, 4)*mu) + (5.0L/24.0L)*sigma*pow(u_2_0, 3)/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*pow(u_2_0, 2)*u_4_0/(pow(dx, 4)*mu) - 3.0L/4.0L*sigma*u_2_0*pow(u_3_0, 2)/(pow(dx, 4)*mu) + (1.0L/4.0L)*sigma*u_2_0*u_3_0*u_4_0/(pow(dx, 4)*mu) - 1.0L/2.0L*sigma*pow(u_3_0, 3)/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*pow(u_3_0, 2)*u_4_0/(pow(dx, 4)*mu);
  J(0,4) = (1.0L/24.0L)*sigma*pow(u_2_0, 3)/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*pow(u_2_0, 2)*u_3_0/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*u_2_0*pow(u_3_0, 2)/(pow(dx, 4)*mu) + (1.0L/24.0L)*sigma*pow(u_3_0, 3)/(pow(dx, 4)*mu);
  return J;
}

//! Return the Jacobian (Non-linear terms are replaced by estimates using the states at the previous time step)
/*
 */
StencilJacobian System::JLinear(const StencilArray& /*states_new*/, const StencilArray& states_old,
            const real dx, const real x, const real t) const
{
  const real sigma = m_sigma;
  const real mu = m_mu;
  const real tau = m_tau(t,x);

  // F should be linear in states_new. J is therefore not dependent on states_new

  const real u0_0_0 = states_old[0][0];
  const real u0_1_0 = states_old[1][0];
  const real u0_2_0 = states_old[2][0];
  const real u0_3_0 = states_old[3][0];
  const real u0_4_0 = states_old[4][0];

  StencilJacobian J_lin;
  J_lin(0,0) = (1.0L/24.0L)*sigma*pow(u0_1_0, 3)/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*pow(u0_1_0, 2)*u0_2_0/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*u0_1_0*pow(u0_2_0, 2)/(pow(dx, 4)*mu) + (1.0L/24.0L)*sigma*pow(u0_2_0, 3)/(pow(dx, 4)*mu);
  J_lin(0,1) = -1.0L/8.0L*sigma*pow(u0_1_0, 3)/(pow(dx, 4)*mu) - 3.0L/8.0L*sigma*pow(u0_1_0, 2)*u0_2_0/(pow(dx, 4)*mu) - 3.0L/8.0L*sigma*u0_1_0*pow(u0_2_0, 2)/(pow(dx, 4)*mu) - 1.0L/6.0L*sigma*pow(u0_2_0, 3)/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*pow(u0_2_0, 2)*u0_3_0/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*u0_2_0*pow(u0_3_0, 2)/(pow(dx, 4)*mu) - 1.0L/24.0L*sigma*pow(u0_3_0, 3)/(pow(dx, 4)*mu);
  J_lin(0,2) = (1.0L/8.0L)*sigma*pow(u0_1_0, 3)/(pow(dx, 4)*mu) + (3.0L/8.0L)*sigma*pow(u0_1_0, 2)*u0_2_0/(pow(dx, 4)*mu) + (3.0L/8.0L)*sigma*u0_1_0*pow(u0_2_0, 2)/(pow(dx, 4)*mu) + (1.0L/4.0L)*sigma*pow(u0_2_0, 3)/(pow(dx, 4)*mu) + (3.0L/8.0L)*sigma*pow(u0_2_0, 2)*u0_3_0/(pow(dx, 4)*mu) + (3.0L/8.0L)*sigma*u0_2_0*pow(u0_3_0, 2)/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*pow(u0_3_0, 3)/(pow(dx, 4)*mu);
  J_lin(0,3) = -1.0L/24.0L*sigma*pow(u0_1_0, 3)/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*pow(u0_1_0, 2)*u0_2_0/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*u0_1_0*pow(u0_2_0, 2)/(pow(dx, 4)*mu) - 1.0L/6.0L*sigma*pow(u0_2_0, 3)/(pow(dx, 4)*mu) - 3.0L/8.0L*sigma*pow(u0_2_0, 2)*u0_3_0/(pow(dx, 4)*mu) - 3.0L/8.0L*sigma*u0_2_0*pow(u0_3_0, 2)/(pow(dx, 4)*mu) - 1.0L/8.0L*sigma*pow(u0_3_0, 3)/(pow(dx, 4)*mu);
  J_lin(0,4) = (1.0L/24.0L)*sigma*pow(u0_2_0, 3)/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*pow(u0_2_0, 2)*u0_3_0/(pow(dx, 4)*mu) + (1.0L/8.0L)*sigma*u0_2_0*pow(u0_3_0, 2)/(pow(dx, 4)*mu) + (1.0L/24.0L)*sigma*pow(u0_3_0, 3)/(pow(dx, 4)*mu);
  return J_lin;
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
