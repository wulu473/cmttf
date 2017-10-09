
#ifndef ROOTFINDER_H_
#define ROOTFINDER_H_

#include <Eigen/Sparse>

#include "Attributes.hpp"
#include "ModuleBase.hpp"

class RootFinderConvergenceException : public std::exception
{

};

class RootFinder : public ParameterisedModuleBase
{
  public:
    typedef RootFinderConvergenceException ConvergenceException;

    virtual std::string baseName() const final;
    RootFinder() {};
    virtual ~RootFinder() {};
    virtual void initialiseFromFile() {};

    typedef Eigen::SparseMatrix<real,Eigen::RowMajor> ESpMatRowMaj;
    typedef Eigen::Matrix<real,Eigen::Dynamic,1> EVector;

    virtual void solveSparse(const std::function<void(const EVector&, EVector&)>& f, 
        const std::function<void(const EVector&, ESpMatRowMaj&)>& J, EVector& x) const;


  protected:
};

#endif
