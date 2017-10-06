
#ifndef NEWTONRAPHSON
#define NEWTONRAPHSON
#include <Eigen/Sparse>

#include "Attributes.hpp"

class ConvergenceException : public std::exception
{

};


class NewtonRaphson
{
  public:
    NewtonRaphson() {};

    typedef Eigen::SparseMatrix<real,Eigen::RowMajor> ESpMatRowMaj;
    typedef Eigen::Matrix<real,Eigen::Dynamic,1> EVector;

    void solveSparse(const std::function<void(const EVector&, EVector&)>& f, 
        const std::function<void(const EVector&, ESpMatRowMaj&)>& J, EVector& x) const;
};

#endif
