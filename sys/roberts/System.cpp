
#include <iomanip>

#include "System.hpp"

State System::F(const StencilArray& states, const real dx, const real x, const real t) const
{
  const real beta = m_beta(t,x);
  const real rho = m_rho;
  const real sigma = m_sigma;
  const real mu = m_mu;
  const real tau = m_tau(t,x);
  const real g = m_gMag;
  const real g1 = m_g1;
  const real g2 = m_g2;

  const real u_0_0 = states[0][0];
  const real u_0_1 = states[0][1];
  const real u_1_0 = states[1][0];
  const real u_1_1 = states[1][1];
  const real u_2_0 = states[2][0];
  const real u_2_1 = states[2][1];
  const real u_3_0 = states[3][0];
  const real u_3_1 = states[3][1];
  const real u_4_0 = states[4][0];
  const real u_4_1 = states[4][1];

  State F;
  F[0] = -beta - 1.0L/2.0L*u_1_0*u_1_1/dx + (1.0L/2.0L)*u_3_0*u_3_1/dx;
  F[1] = -0.82245*g*g1*rho + 2.467*mu*u_2_1/pow(u_2_0, 2) - 1.234*tau/u_2_0 + 0.411225*g*g2*rho*u_1_0/dx - 0.411225*g*g2*rho*u_3_0/dx - 0.75205*rho*pow(u_1_0, 0.0985)*u_1_1*pow(u_2_0, -0.0985)*u_2_1/dx + 0.75205*rho*pow(u_2_0, -0.0985)*u_2_1*pow(u_3_0, 0.0985)*u_3_1/dx - 1.02325*mu*pow(u_0_0, -0.143)*u_0_1*pow(u_1_0, 1.466)*pow(u_2_0, -1.323)/pow(dx, 2) - 0.075*mu*pow(u_1_0, 2)*u_2_1/(pow(dx, 2)*pow(u_2_0, 2)) + 0.15*mu*u_1_0*u_2_1*u_3_0/(pow(dx, 2)*pow(u_2_0, 2)) + 1.02325*mu*pow(u_1_0, 1.466)*pow(u_2_0, -1.466)*u_2_1/pow(dx, 2) + 1.02325*mu*pow(u_2_0, -1.466)*u_2_1*pow(u_3_0, 1.466)/pow(dx, 2) - 1.02325*mu*pow(u_2_0, -1.323)*pow(u_3_0, 1.466)*pow(u_4_0, -0.143)*u_4_1/pow(dx, 2) - 0.075*mu*u_2_1*pow(u_3_0, 2)/(pow(dx, 2)*pow(u_2_0, 2)) + 0.411225*sigma*u_0_0/pow(dx, 3) - 0.82245*sigma*u_1_0/pow(dx, 3) + 0.82245*sigma*u_3_0/pow(dx, 3) - 0.411225*sigma*u_4_0/pow(dx, 3);
  return F;
}

StencilJacobian System::J(const StencilArray& states, const real dx, const real x, const real t) const{
  const real rho = m_rho;
  const real sigma = m_sigma;
  const real mu = m_mu;
  const real tau = m_tau(t,x);
  const real g = m_gMag;
  const real g2 = m_g2;

  const real u_0_0 = states[0][0];
  const real u_0_1 = states[0][1];
  const real u_1_0 = states[1][0];
  const real u_1_1 = states[1][1];
  const real u_2_0 = states[2][0];
  const real u_2_1 = states[2][1];
  const real u_3_0 = states[3][0];
  const real u_3_1 = states[3][1];
  const real u_4_0 = states[4][0];
  const real u_4_1 = states[4][1];

  StencilJacobian J;
  J(0,0) = 0;
  J(1,0) = 0.14632475*mu*pow(u_0_0, -1.143)*u_0_1*pow(u_1_0, 1.466)*pow(u_2_0, -1.323)/pow(dx, 2) + 0.411225*sigma/pow(dx, 3);
  J(0,1) = 0;
  J(1,1) = -1.02325*mu*pow(u_0_0, -0.143)*pow(u_1_0, 1.466)*pow(u_2_0, -1.323)/pow(dx, 2);
  J(0,2) = -1.0L/2.0L*u_1_1/dx;
  J(1,2) = 0.411225*g*g2*rho/dx - 0.074076925*rho*pow(u_1_0, -0.9015)*u_1_1*pow(u_2_0, -0.0985)*u_2_1/dx - 1.5000845*mu*pow(u_0_0, -0.143)*u_0_1*pow(u_1_0, 0.466)*pow(u_2_0, -1.323)/pow(dx, 2) + 1.5000845*mu*pow(u_1_0, 0.466)*pow(u_2_0, -1.466)*u_2_1/pow(dx, 2) - 0.15*mu*u_1_0*u_2_1/(pow(dx, 2)*pow(u_2_0, 2)) + 0.15*mu*u_2_1*u_3_0/(pow(dx, 2)*pow(u_2_0, 2)) - 0.82245*sigma/pow(dx, 3);
  J(0,3) = -1.0L/2.0L*u_1_0/dx;
  J(1,3) = -0.75205*rho*pow(u_1_0, 0.0985)*pow(u_2_0, -0.0985)*u_2_1/dx;
  J(0,4) = 0;
  J(1,4) = -4.934*mu*u_2_1/pow(u_2_0, 3) + 1.234*tau/pow(u_2_0, 2) + 0.074076925*rho*pow(u_1_0, 0.0985)*u_1_1*pow(u_2_0, -1.0985)*u_2_1/dx - 0.074076925*rho*pow(u_2_0, -1.0985)*u_2_1*pow(u_3_0, 0.0985)*u_3_1/dx + 1.35375975*mu*pow(u_0_0, -0.143)*u_0_1*pow(u_1_0, 1.466)*pow(u_2_0, -2.323)/pow(dx, 2) + 0.15*mu*pow(u_1_0, 2)*u_2_1/(pow(dx, 2)*pow(u_2_0, 3)) - 0.3*mu*u_1_0*u_2_1*u_3_0/(pow(dx, 2)*pow(u_2_0, 3)) - 1.5000845*mu*pow(u_1_0, 1.466)*pow(u_2_0, -2.466)*u_2_1/pow(dx, 2) - 1.5000845*mu*pow(u_2_0, -2.466)*u_2_1*pow(u_3_0, 1.466)/pow(dx, 2) + 1.35375975*mu*pow(u_2_0, -2.323)*pow(u_3_0, 1.466)*pow(u_4_0, -0.143)*u_4_1/pow(dx, 2) + 0.15*mu*u_2_1*pow(u_3_0, 2)/(pow(dx, 2)*pow(u_2_0, 3));
  J(0,5) = 0;
  J(1,5) = 2.467*mu/pow(u_2_0, 2) - 0.75205*rho*pow(u_1_0, 0.0985)*u_1_1*pow(u_2_0, -0.0985)/dx + 0.75205*rho*pow(u_2_0, -0.0985)*pow(u_3_0, 0.0985)*u_3_1/dx - 0.075*mu*pow(u_1_0, 2)/(pow(dx, 2)*pow(u_2_0, 2)) + 0.15*mu*u_1_0*u_3_0/(pow(dx, 2)*pow(u_2_0, 2)) + 1.02325*mu*pow(u_1_0, 1.466)*pow(u_2_0, -1.466)/pow(dx, 2) + 1.02325*mu*pow(u_2_0, -1.466)*pow(u_3_0, 1.466)/pow(dx, 2) - 0.075*mu*pow(u_3_0, 2)/(pow(dx, 2)*pow(u_2_0, 2));
  J(0,6) = (1.0L/2.0L)*u_3_1/dx;
  J(1,6) = -0.411225*g*g2*rho/dx + 0.074076925*rho*pow(u_2_0, -0.0985)*u_2_1*pow(u_3_0, -0.9015)*u_3_1/dx + 0.15*mu*u_1_0*u_2_1/(pow(dx, 2)*pow(u_2_0, 2)) + 1.5000845*mu*pow(u_2_0, -1.466)*u_2_1*pow(u_3_0, 0.466)/pow(dx, 2) - 1.5000845*mu*pow(u_2_0, -1.323)*pow(u_3_0, 0.466)*pow(u_4_0, -0.143)*u_4_1/pow(dx, 2) - 0.15*mu*u_2_1*u_3_0/(pow(dx, 2)*pow(u_2_0, 2)) + 0.82245*sigma/pow(dx, 3);
  J(0,7) = (1.0L/2.0L)*u_3_0/dx;
  J(1,7) = 0.75205*rho*pow(u_2_0, -0.0985)*u_2_1*pow(u_3_0, 0.0985)/dx;
  J(0,8) = 0;
  J(1,8) = 0.14632475*mu*pow(u_2_0, -1.323)*pow(u_3_0, 1.466)*pow(u_4_0, -1.143)*u_4_1/pow(dx, 2) - 0.411225*sigma/pow(dx, 3);
  J(0,9) = 0;
  J(1,9) = -1.02325*mu*pow(u_2_0, -1.323)*pow(u_3_0, 1.466)*pow(u_4_0, -0.143)/pow(dx, 2);
  return J;
}

//! Return a factor the time derivative is scaled with
State System::factorTimeDeriv() const
{
  State factor;
  factor[0] = 1.;
  factor[1] = m_rho;
  return factor;
}

void System::initialise(const real mu, const real rho, const real g1, const real g2, 
          const real sigma,
          const TimeSpaceDependReal tau, const TimeSpaceDependReal beta)
{
  m_mu = mu;
  m_rho = rho;
  m_tau = tau;
  m_sigma = sigma;
  m_beta = beta;
  m_gMag = sqrt(g1*g1+g2*g2);
  m_g1 = g1;
  m_g2 = g2;
  if(m_gMag > 0)
  {
    m_g1 /= m_gMag;
    m_g2 /= m_gMag;
  }
}

void System::initialiseFromFile()
{
  const real mu = getParameter<real>("mu");
  const real rho = getParameter<real>("rho");
  const TimeSpaceDependReal tau = getParameter<TimeSpaceDependReal>("tau");
  const TimeSpaceDependReal beta = getParameter<TimeSpaceDependReal>("beta");
  const real sigma = getParameter<real>("sigma");
  const std::vector<real> g = getVectorParameter<real>("g");

  assert(g.size() == 2); // Check if g is a 2D vector in the settings file

  this->initialise(mu,rho,g[0],g[1],sigma,tau,beta);
}

bool System::checkValid(Ref<State> state) const
{
  if(state[0] <= 0.)
  { state[0] = std::numeric_limits<real>::epsilon();
  }
  return true;
}
