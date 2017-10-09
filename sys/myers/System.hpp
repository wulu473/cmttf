
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
    State FLinear(const StencilArray& states, const StencilArray& states_old,
                              const real dx, const real x, const real t) const;

    StencilJacobian J(const StencilArray& states, const real dx, const real x, const real t) const;
    StencilJacobian JLinear(const StencilArray& states, const StencilArray& states_old,
                                        const real dx, const real x, const real t) const;

    void initialise(const real mu, const real sigma, const TimeSpaceDependReal tau, 
                    const TimeSpaceDependReal beta);

    void initialiseFromFile();

  private:
    // Space and time dependent parameters
    
    //! Incoming water 
    TimeSpaceDependReal m_beta;

    //! Shear stress 
    TimeSpaceDependReal m_tau;


    // Constants
    
    //! Viscosity of water
    real m_mu;
    
    //! Surface tension of water
    real m_sigma;
};

#endif
