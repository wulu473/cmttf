
#include "Expression.hpp"
#include "ModuleList.hpp"
#include "Domain.hpp"

REGISTERIMPL(Expression);

Expression::Expression()
{

}

Expression::~Expression()
{

}

std::string Expression::moduleName() const
{
  return "Expression";
}

void Expression::setData(std::shared_ptr<DataPatch> data) const
{
  static const std::shared_ptr<const Domain> domain = ModuleList::uniqueModule<Domain>();

  for(unsigned int cell=0;cell<data->rows();cell++)
  {
    const real s = domain->s(cell);
    for(unsigned int i=0; i<SystemAttributes::stateSize;i++)
    {
      (*data)(cell,i) = m_state[i](0.,s); // t = 0.
    }
  }
}

void Expression::initialise(const StateExpr& state)
{
  m_state = state;
}

void Expression::initialiseFromParameters(const Parameters& params)
{
  std::vector<TimeSpaceDependReal> stateExprVec = getVectorParameter<TimeSpaceDependReal>(params, "state");
  assert(SystemAttributes::stateSize == stateExprVec.size());
  for(unsigned int i=0; i < SystemAttributes::stateSize; i++)
  {
    m_state[i] = stateExprVec[i];
  }
}

