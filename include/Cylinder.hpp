
#ifndef CYLINDER_H_
#define CYLINDER_H_

#include "Domain.hpp"

//! Define a cylindrical substrate
/** A cylindrical substrate s=0 maps to (R,0) and the positive
 *  s direction is counter clockwise
 *
 */
class Cylinder : public Domain
{
  REGISTER(Cylinder);
  public:
    virtual std::string moduleName() const;
    Cylinder();
    virtual ~Cylinder();

    virtual Coord x(const real s) const;

    virtual Coord x_(const unsigned int i, const real s) const;

    virtual real kappa(const real s) const;
    virtual real dkappa_ds(const real s) const;

    virtual void initialiseFromParameters(const Parameters& params);
    void initialise(const int N, const real thetaMin, const real thetaMax, const real radius);
  protected:
    real m_radius;
  private:
};

#endif
