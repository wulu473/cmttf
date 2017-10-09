
#ifndef SYSTEM_H_
#define SYSTEM_H_

#include <vector>
#include <Eigen/Sparse>

#include "Attributes.hpp"
#include "SystemBase.hpp"

class System : public SystemBase
{
  public:
    State F(const StencilArray& states, const real dx, const real x, const real t) const;
    State FLinear(const StencilArray& states, const StencilArray&,
                  const real dx, const real x, const real t) const {throw NotImplemented();}

    StencilJacobian J(const StencilArray& states, const real dx, const real x, const real t) const;
    StencilJacobian JLinear(const StencilArray& states, const StencilArray&,
                            const real dx, const real x, const real t) const {throw NotImplemented();}

    void initialise(const real mu, const real rho, const real g1, const real g2, const real sigma,
                    const TimeSpaceDependReal tau, const TimeSpaceDependReal beta);

    void initialiseFromFile();

  private:
    // Space and time dependent parameters

    //! Incoming water 
    TimeSpaceDependReal m_beta;

    //! Shear stress 
    TimeSpaceDependReal m_tau;

    //! Viscosity of water
    real m_mu;
    
    //! Density of water
    real m_rho;
    
    //! Surface tension of water
    real m_sigma;
    
    //! Magnitude of gravity vector
    real m_gMag;
    
    //! Direction of gravity (has to be normalised)
    real m_g1,m_g2;
};

#endif
