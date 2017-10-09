
#ifndef NEWTONTRUSTREGION_H_
#define NEWTONTRUSTREGION_H_

#include "RootFinder.hpp"

class NewtonTrustRegion : public RootFinder
{
  REGISTER(NewtonTrustRegion);
  public:
    std::string moduleName() const;
    NewtonTrustRegion() {};

    virtual void solveSparse(const std::function<void(const EVector&, EVector&)>& f, 
        const std::function<void(const EVector&, ESpMatRowMaj&)>& J, EVector& x) const;
};

#endif
