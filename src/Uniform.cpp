
#include "Uniform.hpp"

REGISTERIMPL(Uniform);

Uniform::Uniform()
{

}

Uniform::~Uniform()
{

}

std::string Uniform::moduleName() const
{
  return "Uniform";
}

void Uniform::setData(std::shared_ptr<DataPatch> data) const
{
  for(unsigned int cell=0;cell<data->rows();cell++)
  {
    for(unsigned int i=0; i<SystemAttributes::stateSize;i++)
    {
      (*data)(cell,i) = m_state[i];
    }
  }
}

void Uniform::initialise(const State& state)
{
  m_state = state;
}

void Uniform::initialiseFromFile()
{
  initialise(
          //! Uniform state
          getParameter<State>("state")
  );
}
