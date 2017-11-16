
#ifndef SYSTEM_H_
#define SYSTEM_H_

#include <vector>
#include <Eigen/Sparse>

#include "Attributes.hpp"
#include "SystemBase.hpp"

class System : public SystemBase
{
  public:
    virtual State F(const StencilArray& states, const real dx, const real x, const real t) const;
    virtual State FLinear(const StencilArray& states, const StencilArray& states_old,
                          const real dx, const real x, const real t) const;

    virtual StencilJacobian J(const StencilArray& states, const real dx,
                              const real x, const real t) const;
    virtual StencilJacobian JLinear(const StencilArray& states, const StencilArray& states_old,
                                        const real dx, const real x, const real t) const;

    void initialise(const real mu, const real sigma, const real g1, const real g2,
                    const TimeSpaceDependReal tau, const TimeSpaceDependReal beta);

    void initialiseFromFile();

    virtual bool checkValid(Ref<State> state) const;

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

    //! Gravity
    real m_g1, m_g2;
};

#endif
