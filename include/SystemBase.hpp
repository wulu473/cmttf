
#ifndef SYSTEMBASE_H_
#define SYSTEMBASE_H_

#include "ModuleBase.hpp"

class SystemBase : public ParameterisedModuleBase
{
  public:
    virtual std::string name() const;
    virtual int stencilSize() const;
};

#endif
