
#ifndef EXPRESSION_H_
#define EXPRESSION_H_

#include "InitialCondition.hpp"

class Expression : public InitialCondition
{
  REGISTER(Expression);
  public:
    typedef Eigen::Matrix<std::function<real(real,real)>,SystemAttributes::stateSize,1> StateExpr;
    virtual std::string moduleName() const;
    Expression();
    virtual ~Expression();

    virtual void setData(std::shared_ptr<DataPatch>) const;

    virtual void initialise(const StateExpr& state);
    virtual void initialiseFromFile();
  protected:
  private:
    StateExpr m_state;
};

#endif
