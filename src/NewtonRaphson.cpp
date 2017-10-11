
#include "NewtonRaphson.hpp"

#include <boost/log/trivial.hpp>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <cublas_v2.h>

#include <helper_functions.h>
#include <helper_cuda.h>

REGISTERIMPL(NewtonRaphson);

std::string NewtonRaphson::moduleName() const
{
  return "NewtonRaphson";
}

/**
 * x [in,out] Guess and result
 */
void NewtonRaphson::solveSparse(const std::function<void(const EVector&, EVector&)>& fun, 
        const std::function<void(const EVector&, ESpMatRowMaj&)>& jac, EVector& x) const
{
  const unsigned int maxIter = 100;
  const real tol = 1.e-12;

  const unsigned int N = x.size();
  const int rowsA = N;
  const int colsA = N;

  ESpMatRowMaj A(N,N);
  EVector f(N);
  EVector delta_x(N);

  fun(x,f);
  const real f02 = f.squaredNorm();

  // Set up CUDA
  cusolverSpHandle_t cusolverHandle = NULL;
  cublasHandle_t cublasHandle = NULL;
  cusparseMatDescr_t descrA = NULL;

  checkCudaErrors(cusolverSpCreate(&cusolverHandle));
  checkCudaErrors(cusparseCreateMatDescr(&descrA));
  checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

  // Prepare device memory
  int *d_csrRowPtrA = NULL;
  int *d_csrColIndA = NULL;
  real *d_csrValA = NULL;
  real *d_x = NULL; // x = A \ b
  real *d_b = NULL; // a copy of h_b

  // d_csrValA's length depends on the number of zeros of A so we can't allocate memory yet
  // d_csrColIndA's length depends on the number of zeros of A so we can't allocate memory yet
  checkCudaErrors(cudaMalloc((void **)&d_csrRowPtrA, sizeof(int)*(rowsA+1)));
  checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(real)*colsA));
  checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(real)*rowsA));

  unsigned int i=0;
  bool converged = false; 
  do 
  {
    // Compute guess for x
    jac(x,A);
    A.makeCompressed(); // Not sure if needed might be a performance hit

    fun(x,f);

#ifdef DEBUGNOTACTIVE
    {
      BOOST_LOG_TRIVIAL(debug) << "Checking for singularity (This might take a very long time): i = " << i;
      // Do a few tests to check if this matrix is invertible
      Eigen::MatrixXd ADense(A);
      BOOST_LOG_TRIVIAL(debug) << "det(A) = " << ADense.determinant();
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(ADense);
      double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
      BOOST_LOG_TRIVIAL(debug) << "kappa(A) = " << cond;
    }
#endif

    int nnzA = A.nonZeros();

    real *h_csrValA = A.valuePtr();

    f *= -1; // h_b points to -f @TODO Use cusparseDcsrmv at some point for this
    real *h_b = f.data();
    int *h_csrColIndA = A.innerIndexPtr();
    int *h_csrRowPtrA = A.outerIndexPtr();

    int reorder = 0; // no reordering
    int singularity = 0; // -1 if A is invertible under tol.

    // Prepare data on device
    checkCudaErrors(cudaMalloc((void **)&d_csrColIndA, sizeof(int)*nnzA));
    checkCudaErrors(cudaMalloc((void **)&d_csrValA   , sizeof(real)*nnzA));

    checkCudaErrors(
        cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, sizeof(int)*(rowsA+1), cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int)*nnzA     , cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(d_csrValA   , h_csrValA   , sizeof(real)*nnzA  , cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(d_b, h_b, sizeof(real)*rowsA, cudaMemcpyHostToDevice));

    // Solve the system
    checkCudaErrors(cusolverSpDcsrlsvqr(
          cusolverHandle, rowsA, nnzA,
          descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
          d_b, tol, reorder, d_x, &singularity));

    // Copy dx results back into b
    checkCudaErrors(cudaMemcpy(delta_x.data(), d_x, sizeof(real)*rowsA, cudaMemcpyDeviceToHost));

    if (d_csrValA   ) { checkCudaErrors(cudaFree(d_csrValA)); }
    if (d_csrColIndA) { checkCudaErrors(cudaFree(d_csrColIndA)); }

    if(0 <= singularity)
    {
      BOOST_LOG_TRIVIAL(error) << "The matrix is singular at row " << singularity << "  under tol (" << tol << ")";
      throw ConvergenceException();
    }

    //  x_(n+1) = x_(n) + dx
    x += delta_x;


    // Check for convergence
    // Use criteria by Knoll(2004)

    // Check residual
    fun(x,f);
    const real res2 = f.squaredNorm()/f02;

#ifdef DEBUG
    BOOST_LOG_TRIVIAL(debug) << "Iteration: " << i << " res = " << sqrt(res2);
#endif

    // Check newton update
    // If < eps^2 additional steps are not going to give more accuracy
    const real dxRel2 = delta_x.squaredNorm()/x.squaredNorm();

    // Should give us approx 1e-8 accuracy
    if( res2 < 1e-24 || dxRel2 < 1e-24 )
    {
      converged = true;  
    }

    i++;
  } while(!converged && i < maxIter);

  // Free device memory
  if (d_csrRowPtrA) { checkCudaErrors(cudaFree(d_csrRowPtrA)); }
  if (d_x) { checkCudaErrors(cudaFree(d_x)); }
  if (d_b) { checkCudaErrors(cudaFree(d_b)); }

  // Tidy up CUDA before any exception is thrown
  if (cusolverHandle) { checkCudaErrors(cusolverSpDestroy(cusolverHandle)); }
  if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
  if (descrA) { checkCudaErrors(cusparseDestroyMatDescr(descrA)); }

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

