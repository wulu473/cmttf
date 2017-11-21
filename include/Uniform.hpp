
#ifndef UNIFORM_H_
#define UNIFORM_H_

#include "InitialCondition.hpp"

class Uniform : public InitialCondition
{
  REGISTER(Uniform);
  public:
    virtual std::string moduleName() const;
    Uniform();
    virtual ~Uniform();

    virtual void setData(std::shared_ptr<DataPatch>) const;

    virtual void initialise(const State& state);
    virtual void initialiseFromParameters(const Parameters&);
  protected:
  private:
    State m_state;
};

#endif
