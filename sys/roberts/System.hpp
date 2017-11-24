
#ifndef SYSTEM_H_
#define SYSTEM_H_
#define ROBERTS

#include <vector>
#include <Eigen/Sparse>

#include "Attributes.hpp"
#include "SystemBase.hpp"

class System : public SystemBase
{
  public:
    typedef SystemAttributes::VariableNames VariableNames;

    virtual State F(const StencilArray& states, const real dx, const real x, const real t) const;
    virtual State FLinear(const StencilArray& /*states_new*/, const StencilArray& /*states_old*/,
                  const real /*dx*/, const real /*x*/, const real /*t*/) const
        {throw NotImplemented();}

    virtual StencilJacobian J(const StencilArray& states, const real dx, const real x, const real t) const;
    virtual StencilJacobian JLinear(const StencilArray& /*states_new*/, const StencilArray& /*states_old*/,
                            const real /*dx*/, const real /*x*/, const real /*t*/) const
        {throw NotImplemented();}

    virtual State factorTimeDeriv() const;

    void initialise(const real mu, const real rho, const real g1, const real g2, const real sigma,
                    const TimeSpaceDependReal tau, const TimeSpaceDependReal beta);

    void initialiseFromParameters(const Parameters& params);

    virtual bool checkValid(Ref<State> state) const;

    // Output related functions
    //
    //! Define names of state elements
    virtual VariableNames variableNames() const;
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
    Coord m_g;
};

#endif
