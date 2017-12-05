
#include "NewtonRaphson_CUDA.hpp"

#include <boost/log/trivial.hpp>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <cublas_v2.h>

#include <helper_functions.h>
#include <helper_cuda.h>


REGISTERIMPL(NewtonRaphson_CUDA);

std::string NewtonRaphson_CUDA::moduleName() const
{
  return "NewtonRaphson_CUDA";
}

/**
 * x [in,out] Guess and result
 */
void NewtonRaphson_CUDA::solveSparse(const std::function<void(const EVector&, EVector&)>& fun, 
        const std::function<void(const EVector&, ESpMatRowMaj&)>& jac, EVector& x,
        const std::function<bool(EVector&)>& restrictDomain) const
{
#ifdef LOGTIMINGS
  BOOST_LOG_TRIVIAL(warning) << "LOGTIMINGS is active. Deactivated for production runs.";
#endif

  const unsigned int maxIter = 100;
  const real tol = 1.e-12;

  int singularity = 0; // -1 if A is invertible under tol.

  const unsigned int N = x.size();
  const int rowsA = N;
  const int colsA = N;

  ESpMatRowMaj A(N,N);
  EVector f(N);
  EVector dx(N);

  fun(x,f);
  const real f02 = f.squaredNorm();

  // Set up CUDA
  cusolverSpHandle_t cusolverHandle = NULL;
  cublasHandle_t cublasHandle = NULL;
  cusparseMatDescr_t descrA = NULL;

  if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS)
  {
    BOOST_LOG_TRIVIAL(error) << "CUBLAS initialization error";
    return exit(34);
  }

  checkCudaErrors(cusolverSpCreate(&cusolverHandle));
  checkCudaErrors(cusparseCreateMatDescr(&descrA));
  checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

  // Prepare device memory
  int *d_csrRowPtrA = NULL;
  int *d_csrColIndA = NULL;
  real *d_csrValA = NULL;
  real *d_x = NULL;
  real *d_dx = NULL;
  real *d_f = NULL;

  // d_csrValA's length depends on the number of zeros of A so we can't allocate memory yet
  // d_csrColIndA's length depends on the number of zeros of A so we can't allocate memory yet
  checkCudaErrors(cudaMalloc((void **)&d_csrRowPtrA, sizeof(int)*(rowsA+1)));
  checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(real)*colsA));
  checkCudaErrors(cudaMalloc((void **)&d_dx, sizeof(real)*colsA));
  checkCudaErrors(cudaMalloc((void **)&d_f, sizeof(real)*rowsA));

  unsigned int i=0;
  bool converged = false; 
  bool singular = false;
  do 
  {

    // Compute guess for x
#ifdef LOGTIMINGS
    BOOST_LOG_TRIVIAL(debug) << "Starting: Setting up function and matrix";
#endif
    jac(x,A);
    A.makeCompressed(); // Not sure if needed might be a performance hit
    fun(x,f);
#ifdef LOGTIMINGS
    BOOST_LOG_TRIVIAL(debug) << "Finished: Setting up function and matrix";
#endif

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

    // Use same naming convention as CUDA samples
    int nnzA = A.nonZeros();
    real *h_csrValA = A.valuePtr();
    int *h_csrColIndA = A.innerIndexPtr();
    int *h_csrRowPtrA = A.outerIndexPtr();

    int reorder = 0; // no reordering
    singularity = 0;

    // Prepare data on device
#ifdef LOGTIMINGS
    BOOST_LOG_TRIVIAL(debug) << "Starting: Allocate memory and transfer data to device";
#endif
    checkCudaErrors(cudaMalloc((void **)&d_csrColIndA, sizeof(int)*nnzA));
    checkCudaErrors(cudaMalloc((void **)&d_csrValA   , sizeof(real)*nnzA));

    checkCudaErrors(
        cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, sizeof(int)*(rowsA+1), cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int)*nnzA     , cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(d_csrValA   , h_csrValA   , sizeof(real)*nnzA  , cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(d_f, f.data(), sizeof(real)*rowsA, cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(d_x, x.data(), sizeof(real)*colsA, cudaMemcpyHostToDevice));
#ifdef LOGTIMINGS
    BOOST_LOG_TRIVIAL(debug) << "Finished: Allocate memory and transfer data to device";
#endif

    // RHS = -f (do it in place to save memory)
    const real negOne = -1.;
    checkCudaErrors(cublasDscal(cublasHandle, rowsA, &negOne, d_f, 1));
    // d_f contains -f from now on

#ifdef LOGTIMINGS
    BOOST_LOG_TRIVIAL(debug) << "Starting: Solve system";
#endif
    // Solve the system
    checkCudaErrors(cusolverSpDcsrlsvqr(
          cusolverHandle, rowsA, nnzA,
          descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
          d_f, tol, reorder, d_dx, &singularity));
#ifdef LOGTIMINGS
    BOOST_LOG_TRIVIAL(debug) << "Finished: Solve system";
#endif

    // Check if A was singular
    singular = (0 <= singularity);

#ifdef LOGTIMINGS
    BOOST_LOG_TRIVIAL(debug) << "Starting: Compute error norms";
#endif
    // Compute norm of increment
    real dx_norm = -1;
    checkCudaErrors(cublasDnrm2(cublasHandle, colsA, d_dx, 1, &dx_norm));

    // Use the norm of old x according to Knoll
    real x_norm = -1;
    checkCudaErrors(cublasDnrm2(cublasHandle, colsA, d_x, 1, &x_norm));

    // x += dx
    const real one = 1.;
    checkCudaErrors(cublasDaxpy(cublasHandle, colsA, &one, d_dx, 1, d_x, 1));

    // Copy d_x results back into x
    checkCudaErrors(cudaMemcpy(x.data(), d_x, sizeof(real)*rowsA, cudaMemcpyDeviceToHost));

    if (d_csrValA   ) { checkCudaErrors(cudaFree(d_csrValA)); }
    if (d_csrColIndA) { checkCudaErrors(cudaFree(d_csrColIndA)); }

    if(!restrictDomain(x))
    {
      BOOST_LOG_TRIVIAL(error) << "A Newton Raphson guess has been found which is not in the expected domain. This usually points to bad starting point for the procedure";
      exit(33); 
    }

    // Check for convergence
    // Use criteria by Knoll(2004)

    // Check residual
    fun(x,f);
    const real res2 = f.squaredNorm()/f02;
#ifdef LOGTIMINGS
    BOOST_LOG_TRIVIAL(debug) << "Finished: Compute error norms";
#endif

#ifdef DEBUG
    BOOST_LOG_TRIVIAL(debug) << "Iteration: " << i << " res = " << sqrt(res2);
#endif

    // Check newton update
    // If < eps^2 additional steps are not going to give more accuracy
    const real dxRel = dx_norm/x_norm;

    // Should give us approx 1e-10 accuracy
    if( res2 < 1e-24 || dxRel < 1e-12 )
    {
      converged = true;  
    }

    i++;
  } while(!converged && i < maxIter && !singular);

  // Free device memory
  if (d_csrRowPtrA) { checkCudaErrors(cudaFree(d_csrRowPtrA)); }
  if (d_x) { checkCudaErrors(cudaFree(d_x)); }
  if (d_f) { checkCudaErrors(cudaFree(d_f)); }
  if (d_dx) { checkCudaErrors(cudaFree(d_dx)); }

  // Tidy up CUDA before any exception is thrown
  if (cusolverHandle) { checkCudaErrors(cusolverSpDestroy(cusolverHandle)); }
  if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
  if (descrA) { checkCudaErrors(cusparseDestroyMatDescr(descrA)); }

  if(singular)
  {
    BOOST_LOG_TRIVIAL(error) << "The matrix is singular at row " << singularity << "  under tol (" << tol << ")";
    throw ConvergenceException();
  }

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

