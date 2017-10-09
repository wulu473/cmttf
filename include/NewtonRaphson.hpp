
#ifndef NEWTONRAPHSON_H_
#define NEWTONRAPHSON_H_

#include "RootFinder.hpp"

class NewtonRaphson : public RootFinder
{
  REGISTER(NewtonRaphson)
  public:
    virtual std::string moduleName() const;
    NewtonRaphson() {}
    virtual ~NewtonRaphson() {}

    virtual void solveSparse(const std::function<void(const EVector&, EVector&)>& f, 
        const std::function<void(const EVector&, ESpMatRowMaj&)>& J, EVector& x) const;

  protected:
};

#endif
