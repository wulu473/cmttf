
#ifndef NEWTONRAPHSONCUDA_H_
#define NEWTONRAPHSONCUDA_H_

#include "RootFinder.hpp"

class NewtonRaphson_CUDA : public RootFinder
{
  REGISTER(NewtonRaphson_CUDA)
  public:
    virtual std::string moduleName() const;
    NewtonRaphson_CUDA() {}
    virtual ~NewtonRaphson_CUDA() {}

    virtual void solveSparse(const std::function<void(const EVector&, EVector&)>& f, 
        const std::function<void(const EVector&, ESpMatRowMaj&)>& J, EVector& x,
        const std::function<bool(EVector&)>& restrictDomain = RootFinder::allValid) const;

  protected:
};

#endif
