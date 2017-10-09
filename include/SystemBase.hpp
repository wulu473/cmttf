
#ifndef SYSTEMBASE_H_
#define SYSTEMBASE_H_

#include "ModuleBase.hpp"

class SystemBase : public ParameterisedModuleBase
{
  public:
    virtual std::string baseName() const final;
    virtual int stencilSize() const;
};

#endif
