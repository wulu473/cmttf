
#ifndef FLAT_H_
#define FLAT_H_

#include "Domain.hpp"

class Flat : public Domain
{
  REGISTER(Flat);
  public:
    virtual std::string moduleName() const;
    Flat();
    virtual ~Flat();

    virtual Coord x(const real s) const;

    virtual Coord x_(const unsigned int i, const real s) const;

    virtual real kappa(const real s) const;
    virtual real dkappa_ds(const real s) const;

    virtual void initialiseFromParameters(const Parameters& params);
    void initialise(const int N, const real L, const real R);
  protected:
  private:
};

#endif
