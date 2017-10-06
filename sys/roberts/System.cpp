
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

  real ST_0_0,ST_0_1,ST_1_0,ST_1_1,ST_2_0,ST_2_1,ST_3_0,ST_3_1,ST_4_0,ST_4_1; 
  ST_0_0 = states[0][0];
  ST_0_1 = states[0][1];
  ST_1_0 = states[1][0];
  ST_1_1 = states[1][1];
  ST_2_0 = states[2][0];
  ST_2_1 = states[2][1];
  ST_3_0 = states[3][0];
  ST_3_1 = states[3][1];
  ST_4_0 = states[4][0];
  ST_4_1 = states[4][1];

  State F;
  F[0] = -1.0L/2.0L*ST_1_0*ST_1_1/dx + (1.0L/2.0L)*ST_3_0*ST_3_1/dx - beta;
  F[1] = -1.02325*pow(ST_0_0, -0.143)*ST_0_1*pow(ST_1_0, 1.466)*pow(ST_2_0, -1.323)*mu/(pow(dx, 2)*rho) + 0.411225*ST_0_0*sigma/(pow(dx, 3)*rho) - 0.75205*pow(ST_1_0, 0.0985)*ST_1_1*pow(ST_2_0, -0.0985)*ST_2_1/dx - 0.075*pow(ST_1_0, 2)*ST_2_1*mu/(pow(ST_2_0, 2)*pow(dx, 2)*rho) + 0.411225*ST_1_0*g*g2/dx - 0.82245*ST_1_0*sigma/(pow(dx, 3)*rho) + 0.15*ST_1_0*ST_2_1*ST_3_0*mu/(pow(ST_2_0, 2)*pow(dx, 2)*rho) + 1.02325*pow(ST_1_0, 1.466)*pow(ST_2_0, -1.466)*ST_2_1*mu/(pow(dx, 2)*rho) + 1.02325*pow(ST_2_0, -1.466)*ST_2_1*pow(ST_3_0, 1.466)*mu/(pow(dx, 2)*rho) - 1.02325*pow(ST_2_0, -1.323)*pow(ST_3_0, 1.466)*pow(ST_4_0, -0.143)*ST_4_1*mu/(pow(dx, 2)*rho) + 0.75205*pow(ST_2_0, -0.0985)*ST_2_1*pow(ST_3_0, 0.0985)*ST_3_1/dx - 0.411225*ST_3_0*g*g2/dx + 0.82245*ST_3_0*sigma/(pow(dx, 3)*rho) - 0.411225*ST_4_0*sigma/(pow(dx, 3)*rho) - 0.82245*g*g1 - 1.234*tau/(ST_2_0*rho) - 0.075*ST_2_1*pow(ST_3_0, 2)*mu/(pow(ST_2_0, 2)*pow(dx, 2)*rho) + 2.467*ST_2_1*mu/(pow(ST_2_0, 2)*rho);
  return F;
}

StencilJacobian System::J(const StencilArray& states, const real dx, const real x, const real t) const{
  const real rho = m_rho;
  const real sigma = m_sigma;
  const real mu = m_mu;
  const real tau = m_tau(t,x);
  const real g = m_gMag;
  const real g2 = m_g2;

  real ST_0_0,ST_0_1,ST_1_0,ST_1_1,ST_2_0,ST_2_1,ST_3_0,ST_3_1,ST_4_0,ST_4_1; 
  ST_0_0 = states[0][0];
  ST_0_1 = states[0][1];
  ST_1_0 = states[1][0];
  ST_1_1 = states[1][1];
  ST_2_0 = states[2][0];
  ST_2_1 = states[2][1];
  ST_3_0 = states[3][0];
  ST_3_1 = states[3][1];
  ST_4_0 = states[4][0];
  ST_4_1 = states[4][1];
  StencilJacobian J;

  J(0,0) = 0;
  J(1,0) = 0.14632475*pow(ST_0_0, -1.143)*ST_0_1*pow(ST_1_0, 1.466)*pow(ST_2_0, -1.323)*mu/(pow(dx, 2)*rho) + 0.411225*sigma/(pow(dx, 3)*rho);
  J(0,1) = 0;
  J(1,1) = -1.02325*pow(ST_0_0, -0.143)*pow(ST_1_0, 1.466)*pow(ST_2_0, -1.323)*mu/(pow(dx, 2)*rho);
  J(0,2) = -1.0L/2.0L*ST_1_1/dx;
  J(1,2) = -1.5000845*pow(ST_0_0, -0.143)*ST_0_1*pow(ST_1_0, 0.466)*pow(ST_2_0, -1.323)*mu/(pow(dx, 2)*rho) - 0.074076925*pow(ST_1_0, -0.9015)*ST_1_1*pow(ST_2_0, -0.0985)*ST_2_1/dx + 1.5000845*pow(ST_1_0, 0.466)*pow(ST_2_0, -1.466)*ST_2_1*mu/(pow(dx, 2)*rho) - 0.15*ST_1_0*ST_2_1*mu/(pow(ST_2_0, 2)*pow(dx, 2)*rho) + 0.411225*g*g2/dx - 0.82245*sigma/(pow(dx, 3)*rho) + 0.15*ST_2_1*ST_3_0*mu/(pow(ST_2_0, 2)*pow(dx, 2)*rho);
  J(0,3) = -1.0L/2.0L*ST_1_0/dx;
  J(1,3) = -0.75205*pow(ST_1_0, 0.0985)*pow(ST_2_0, -0.0985)*ST_2_1/dx;
  J(0,4) = 0;
  J(1,4) = 1.35375975*pow(ST_0_0, -0.143)*ST_0_1*pow(ST_1_0, 1.466)*pow(ST_2_0, -2.323)*mu/(pow(dx, 2)*rho) + 0.074076925*pow(ST_1_0, 0.0985)*ST_1_1*pow(ST_2_0, -1.0985)*ST_2_1/dx + 0.15*pow(ST_1_0, 2)*ST_2_1*mu/(pow(ST_2_0, 3)*pow(dx, 2)*rho) - 0.3*ST_1_0*ST_2_1*ST_3_0*mu/(pow(ST_2_0, 3)*pow(dx, 2)*rho) - 1.5000845*pow(ST_1_0, 1.466)*pow(ST_2_0, -2.466)*ST_2_1*mu/(pow(dx, 2)*rho) - 1.5000845*pow(ST_2_0, -2.466)*ST_2_1*pow(ST_3_0, 1.466)*mu/(pow(dx, 2)*rho) + 1.35375975*pow(ST_2_0, -2.323)*pow(ST_3_0, 1.466)*pow(ST_4_0, -0.143)*ST_4_1*mu/(pow(dx, 2)*rho) - 0.074076925*pow(ST_2_0, -1.0985)*ST_2_1*pow(ST_3_0, 0.0985)*ST_3_1/dx + 1.234*tau/(pow(ST_2_0, 2)*rho) + 0.15*ST_2_1*pow(ST_3_0, 2)*mu/(pow(ST_2_0, 3)*pow(dx, 2)*rho) - 4.934*ST_2_1*mu/(pow(ST_2_0, 3)*rho);
  J(0,5) = 0;
  J(1,5) = -0.75205*pow(ST_1_0, 0.0985)*ST_1_1*pow(ST_2_0, -0.0985)/dx - 0.075*pow(ST_1_0, 2)*mu/(pow(ST_2_0, 2)*pow(dx, 2)*rho) + 0.15*ST_1_0*ST_3_0*mu/(pow(ST_2_0, 2)*pow(dx, 2)*rho) + 1.02325*pow(ST_1_0, 1.466)*pow(ST_2_0, -1.466)*mu/(pow(dx, 2)*rho) + 1.02325*pow(ST_2_0, -1.466)*pow(ST_3_0, 1.466)*mu/(pow(dx, 2)*rho) + 0.75205*pow(ST_2_0, -0.0985)*pow(ST_3_0, 0.0985)*ST_3_1/dx - 0.075*pow(ST_3_0, 2)*mu/(pow(ST_2_0, 2)*pow(dx, 2)*rho) + 2.467*mu/(pow(ST_2_0, 2)*rho);
  J(0,6) = (1.0L/2.0L)*ST_3_1/dx;
  J(1,6) = 0.15*ST_1_0*ST_2_1*mu/(pow(ST_2_0, 2)*pow(dx, 2)*rho) + 1.5000845*pow(ST_2_0, -1.466)*ST_2_1*pow(ST_3_0, 0.466)*mu/(pow(dx, 2)*rho) - 1.5000845*pow(ST_2_0, -1.323)*pow(ST_3_0, 0.466)*pow(ST_4_0, -0.143)*ST_4_1*mu/(pow(dx, 2)*rho) + 0.074076925*pow(ST_2_0, -0.0985)*ST_2_1*pow(ST_3_0, -0.9015)*ST_3_1/dx - 0.411225*g*g2/dx + 0.82245*sigma/(pow(dx, 3)*rho) - 0.15*ST_2_1*ST_3_0*mu/(pow(ST_2_0, 2)*pow(dx, 2)*rho);
  J(0,7) = (1.0L/2.0L)*ST_3_0/dx;
  J(1,7) = 0.75205*pow(ST_2_0, -0.0985)*ST_2_1*pow(ST_3_0, 0.0985)/dx;
  J(0,8) = 0;
  J(1,8) = 0.14632475*pow(ST_2_0, -1.323)*pow(ST_3_0, 1.466)*pow(ST_4_0, -1.143)*ST_4_1*mu/(pow(dx, 2)*rho) - 0.411225*sigma/(pow(dx, 3)*rho);
  J(0,9) = 0;
  J(1,9) = -1.02325*pow(ST_2_0, -1.323)*pow(ST_3_0, 1.466)*pow(ST_4_0, -0.143)*mu/(pow(dx, 2)*rho);
  return J;
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
