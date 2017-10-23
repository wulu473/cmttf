
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
    typedef Eigen::SparseMatrix<real,Eigen::RowMajor> ESpMatRowMaj;
    typedef Eigen::Matrix<real,Eigen::Dynamic,1> EVector;

    typedef RootFinderConvergenceException ConvergenceException;

    virtual std::string baseName() const final;
    RootFinder() {};
    virtual ~RootFinder() {};
    virtual void initialiseFromFile() {};

    //! Indicate that all guesses are valid
    static bool allValid(EVector&);

    //! Find the root of a sparse system of nonlinear equations
    /*
     * f [in] Function which root is to be found
     * J [in] Jacobian of f
     * x [in,out] Initial guess and if successful returns root
     * restricDomain [in] A function that takes a guess and may change guesses to
     *                    restrict it to a defined domain. It has to return true
     *                    if it was successful
     */
    virtual void solveSparse(const std::function<void(const EVector&, EVector&)>& f, 
        const std::function<void(const EVector&, ESpMatRowMaj&)>& J, EVector& x,
        const std::function<bool(EVector&)>& restrictDomain = allValid) const;


  protected:
};

#endif
