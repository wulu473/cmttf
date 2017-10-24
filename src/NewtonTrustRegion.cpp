
#include "NewtonTrustRegion.hpp"

#include <Eigen/SparseQR>

#include <boost/log/trivial.hpp>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <cublas_v2.h>

#include <helper_functions.h>
#include <helper_cuda.h>

REGISTERIMPL(NewtonTrustRegion);

std::string NewtonTrustRegion::moduleName() const
{
  return "NewtonTrustRegion";
}

/**
 * x [in,out] Guess and result
 */
void NewtonTrustRegion::solveSparse(const std::function<void(const EVector&, EVector&)>& fun, 
        const std::function<void(const EVector&, ESpMatRowMaj&)>& jac, EVector& x,
        const std::function<bool(EVector&)>& restrictDomain) const
{
  const unsigned int N = x.size();

  // Algorithm 11.4 from Numerical Optimization by J. Nocedal, S. J. Wright
  // using the dog leg method

  const unsigned int maxIter = 10000;

  const real deltaMax = 1e5*x.norm();
  const real eta = 0.1;

  real delta = deltaMax/4; // trust region size


  ESpMatRowMaj A(N,N);
  EVector f(N);
  EVector delta_x(N);

  fun(x,f);
  const real f02 = f.squaredNorm();

  unsigned int i=0;
  bool converged = false; 
  do 
  {
    fun(x,f);
    jac(x,A);

    EVector p_k;

    const real tau_k_denom = delta*f.transpose()*A*A.transpose()*A*A.transpose()*f;
    const real tau_k = std::min(1.,pow((A.transpose()*f).norm(),3)/tau_k_denom);

    const EVector pC_k = -tau_k*(delta/(A.transpose()*f).norm())*A.transpose()*f;

    if(fabs(pC_k.squaredNorm() - delta*delta) < std::numeric_limits<real>::epsilon())
    {
      //
      p_k = pC_k; 
    }
    else
    {
      Eigen::SparseLU<ESpMatRowMaj> solver;
      solver.compute(A);
      if(solver.info() != Eigen::Success)
      {
        BOOST_LOG_TRIVIAL(error) << "The matrix could not be decomposed";
        throw ConvergenceException();
      }
      const EVector pJ_k = solver.solve(-f);
      if(solver.info() != Eigen::Success)
      {
        BOOST_LOG_TRIVIAL(error) << "Solving failed";
        throw ConvergenceException();
      }

      real tauL = 0.; 
      real tauR = 1.; 
      do
      {
        const real tau = 0.5*(tauL+tauR);
        const real pKNorm = (pC_k + tau*(pJ_k-pC_k)).norm();
        if (pKNorm < delta)
        {
          tauL = tau;
        }
        else
        {
          tauR = tau;
        }
      } while ( tauR - tauL > 1e-8 );
      const real tau = std::min(1.,0.5*(tauL+tauR));
    

      p_k = pC_k + tau*( pJ_k - pC_k);
    }


    EVector fNew = f; // Hopefully this copies
    EVector xNew = x+p_k;
    if(!restrictDomain(xNew))
    {
      BOOST_LOG_TRIVIAL(error) << "A guess has been found which is not in the expected domain. This usually points to bad starting point for the procedure";
      exit(33); // Consider throwing exception instead
    }
    p_k = xNew - x;
    fun(xNew,fNew);
    const real rho_k_denom = f.squaredNorm() - (f + A*p_k).squaredNorm();
    const real rho_k = (f.squaredNorm() - fNew.squaredNorm())/rho_k_denom;

#ifdef DEBUG
      BOOST_LOG_TRIVIAL(debug) << "Metric for approximation rho = " << rho_k;
#endif

    if (rho_k < 1./4.)
    {
      delta = 1./4.*p_k.norm();
#ifdef DEBUG
      BOOST_LOG_TRIVIAL(debug) << "The model function is not a good approximation. Reducing trust region to delta = " << delta;
#endif
    }
    else
    {
      if(rho_k > 3./4. && fabs(p_k.norm() - delta) < sqrt(std::numeric_limits<real>::epsilon()))
      {
        delta = std::min(2*delta, deltaMax);
#ifdef DEBUG
        if(fabs(delta - deltaMax) < std::numeric_limits<real>::epsilon())
        {
          BOOST_LOG_TRIVIAL(debug) << "Trust region size has reached maximum size. Consider increasing value of deltaMax";
        }
        else
        {
          BOOST_LOG_TRIVIAL(debug) << "The model function is an excellent approximation. Increasing trust region size to delta = " << delta; 
        }
#endif
      }
      else
      {
        // delta_{n+1} = delta_{n}
#ifdef DEBUG
        BOOST_LOG_TRIVIAL(debug) << "The model function is a good approximation. Keeping trust region size at delta = " << delta;
#endif
      }
    }

    real dxRel = std::numeric_limits<real>::max();
    if (rho_k > eta)
    {
      dxRel = p_k.norm()/x.norm();
      x = xNew;
    }
    else
    {
      // x_{n+1} = x_{n}
    }

    // Check for convergence
    // Use criteria by Knoll(2004)

    // Check residual
    fun(x,f);
    const real res2 = f.squaredNorm()/f02;


#ifdef DEBUG
    BOOST_LOG_TRIVIAL(debug) << "NewtonTrustRegion: Iteration: " << i << " res = " << sqrt(res2) << " dxRel = " << dxRel;
#endif

    if( res2 < 1e-24 || dxRel < 1e-12 )
    {
      converged = true;  
    }
   
    i++;
  } while(!converged && i < maxIter);

  if(!converged)
  {
    BOOST_LOG_TRIVIAL(error) << "The solution has not converged after " << i << " iterations. Try a better guess.";
    throw ConvergenceException();
  }
  else
  {
    BOOST_LOG_TRIVIAL(debug) << "Solution has converged after " << i << " iterations.";
  }
}

