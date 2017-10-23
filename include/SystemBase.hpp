
#ifndef SYSTEMBASE_H_
#define SYSTEMBASE_H_

#include "ModuleBase.hpp"

class SystemBase : public ParameterisedModuleBase
{
  public:
    virtual std::string baseName() const final;
    virtual std::string moduleName() const final; // don't allow systems to have other names
    virtual int stencilSize() const;

    //! A factor that is applied to the timer derivative term
    virtual State factorTimeDeriv() const;

    //! Check if a state is valid and try to correct it
    /*
     * state [in,out] State to check and return the correct one
     * success [ret] Flag whether the correction was successful
     */
    virtual bool checkValid(Ref<State> state) const;
};

#endif
