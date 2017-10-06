
#ifndef INITIALCONDITON_H_
#define INITIALCONDITON_H_

#include "ModuleBase.hpp"
#include "DataPatch.hpp"

class InitialCondition : public ParameterisedModuleBase
{
  public:
    virtual std::string name() const;

    InitialCondition();
    virtual ~InitialCondition();

    virtual void setData(std::shared_ptr<DataPatch>) const;
};

#endif
