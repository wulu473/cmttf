
#ifndef IMPLICITLINEARSOLVER_H_
#define IMPLICITLINEARSOLVER_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "TimeIntegrator.hpp"

class ImplicitLinearSolver : public TimeIntegrator
{
  REGISTER(ImplicitLinearSolver)
  public:
    virtual std::string moduleName() const;

    virtual void initialise(const real alpha); 
    virtual void initialiseFromFile();

    typedef Eigen::Matrix<real,Eigen::Dynamic,1> Vector;
    typedef Eigen::SparseMatrix<real,Eigen::RowMajor> SpMatRowMaj;

    void function(const Vector& states_old, const Vector& states_new, const real dt, const real t, Vector& f) const;

    void jacobian(const Vector& states_old, const Vector& states_new, const real dt, const real t, SpMatRowMaj& J) const;

    void advance(std::shared_ptr<DataPatch> states, const real dt, const real t) const;

  protected:
};

#endif
