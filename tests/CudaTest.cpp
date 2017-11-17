
#include <boost/test/unit_test.hpp>

#include <Eigen/Sparse>
#include <cuda_runtime.h>

#include <cusparse.h>
#include <cusolverSp.h>
#include <helper_cuda.h>

#include <vector>
#include <iostream>

#include "Log.hpp"
#include "CUDA.hpp"

#ifdef DISABLE_CUDA
#warning CUDA is disabled. No CUDA tests are run
#else

typedef double real;
typedef Eigen::SparseMatrix<real> SpMat;
typedef Eigen::SparseMatrix<real,Eigen::RowMajor> SpMatRowMaj;
typedef Eigen::Triplet<real> T;

BOOST_AUTO_TEST_SUITE(CUDATest)

BOOST_AUTO_TEST_CASE(FreeMemoryTest)
{
  auto freeMem = CUDA::freeMemory();
  for(auto devFreeMem : freeMem)
  {
    BOOST_LOG_TRIVIAL(info) << "Device " << devFreeMem.first << " free memory: " << devFreeMem.second;
  }
  BOOST_LOG_TRIVIAL(info) << "Device with most free memory: " << CUDA::deviceMostFreeMemory();
}

BOOST_AUTO_TEST_CASE(Nvidia_example_csc)
{
  /*
   * Compare to example on Nvidia website http://docs.nvidia.com/cuda/cusparse/index.html#using-the-cusparse-api
   */
  std::vector<T> coeffs;

  coeffs.push_back(T(0,0,1.0));
  coeffs.push_back(T(0,1,4.0));
  coeffs.push_back(T(1,1,2.0));
  coeffs.push_back(T(1,2,3.0));
  coeffs.push_back(T(2,0,5.0));
  coeffs.push_back(T(2,3,7.0));
  coeffs.push_back(T(2,4,8.0));
  coeffs.push_back(T(3,2,9.0));
  coeffs.push_back(T(3,4,6.0));

  SpMat A(4,5);
  A.setFromTriplets(coeffs.begin(),coeffs.end());

  A.makeCompressed();

  real * cscValA = A.valuePtr();
  int *cscRowIndA = A.innerIndexPtr();
  int *cscColPtrA = A.outerIndexPtr();

  BOOST_CHECK_EQUAL(cscRowIndA[0],0);
  BOOST_CHECK_EQUAL(cscRowIndA[1],2);
  BOOST_CHECK_EQUAL(cscRowIndA[2],0);
  BOOST_CHECK_EQUAL(cscRowIndA[3],1);
  BOOST_CHECK_EQUAL(cscRowIndA[4],1);
  BOOST_CHECK_EQUAL(cscRowIndA[5],3);
  BOOST_CHECK_EQUAL(cscRowIndA[6],2);
  BOOST_CHECK_EQUAL(cscRowIndA[7],2);
  BOOST_CHECK_EQUAL(cscRowIndA[8],3);

  BOOST_CHECK_EQUAL(cscColPtrA[0],0);
  BOOST_CHECK_EQUAL(cscColPtrA[1],2);
  BOOST_CHECK_EQUAL(cscColPtrA[2],4);
  BOOST_CHECK_EQUAL(cscColPtrA[3],6);
  BOOST_CHECK_EQUAL(cscColPtrA[4],7);
  BOOST_CHECK_EQUAL(cscColPtrA[5],9);

  BOOST_CHECK_CLOSE(cscValA[0],1.0,1e-3);
  BOOST_CHECK_CLOSE(cscValA[1],5.0,1e-3);
  BOOST_CHECK_CLOSE(cscValA[2],4.0,1e-3);
  BOOST_CHECK_CLOSE(cscValA[3],2.0,1e-3);
  BOOST_CHECK_CLOSE(cscValA[4],3.0,1e-3);
  BOOST_CHECK_CLOSE(cscValA[5],9.0,1e-3);
  BOOST_CHECK_CLOSE(cscValA[6],7.0,1e-3);
  BOOST_CHECK_CLOSE(cscValA[7],8.0,1e-3);
  BOOST_CHECK_CLOSE(cscValA[8],6.0,1e-3);
}

BOOST_AUTO_TEST_CASE(Nvidia_example_csr)
{
  /*
   * Compare to example on Nvidia website http://docs.nvidia.com/cuda/cusparse/index.html#using-the-cusparse-api
   */
  std::vector<T> coeffs;

  coeffs.push_back(T(0,0,1.0));
  coeffs.push_back(T(0,1,4.0));
  coeffs.push_back(T(1,1,2.0));
  coeffs.push_back(T(1,2,3.0));
  coeffs.push_back(T(2,0,5.0));
  coeffs.push_back(T(2,3,7.0));
  coeffs.push_back(T(2,4,8.0));
  coeffs.push_back(T(3,2,9.0));
  coeffs.push_back(T(3,4,6.0));

  SpMatRowMaj A(4,5);
  A.setFromTriplets(coeffs.begin(),coeffs.end());

  A.makeCompressed();

  real * csrValA = A.valuePtr();
  int *csrColIndA = A.innerIndexPtr();
  int *csrRowPtrA = A.outerIndexPtr();

  BOOST_CHECK_EQUAL(csrColIndA[0],0);
  BOOST_CHECK_EQUAL(csrColIndA[1],1);
  BOOST_CHECK_EQUAL(csrColIndA[2],1);
  BOOST_CHECK_EQUAL(csrColIndA[3],2);
  BOOST_CHECK_EQUAL(csrColIndA[4],0);
  BOOST_CHECK_EQUAL(csrColIndA[5],3);
  BOOST_CHECK_EQUAL(csrColIndA[6],4);
  BOOST_CHECK_EQUAL(csrColIndA[7],2);
  BOOST_CHECK_EQUAL(csrColIndA[8],4);

  BOOST_CHECK_EQUAL(csrRowPtrA[0],0);
  BOOST_CHECK_EQUAL(csrRowPtrA[1],2);
  BOOST_CHECK_EQUAL(csrRowPtrA[2],4);
  BOOST_CHECK_EQUAL(csrRowPtrA[3],7);
  BOOST_CHECK_EQUAL(csrRowPtrA[4],9);

  BOOST_CHECK_CLOSE(csrValA[0],1.0,1e-3);
  BOOST_CHECK_CLOSE(csrValA[1],4.0,1e-3);
  BOOST_CHECK_CLOSE(csrValA[2],2.0,1e-3);
  BOOST_CHECK_CLOSE(csrValA[3],3.0,1e-3);
  BOOST_CHECK_CLOSE(csrValA[4],5.0,1e-3);
  BOOST_CHECK_CLOSE(csrValA[5],7.0,1e-3);
  BOOST_CHECK_CLOSE(csrValA[6],8.0,1e-3);
  BOOST_CHECK_CLOSE(csrValA[7],9.0,1e-3);
  BOOST_CHECK_CLOSE(csrValA[8],6.0,1e-3);
}

BOOST_AUTO_TEST_CASE(QR)
{
  const int colsA = 4; // number of columns of A
  const int rowsA = colsA; // number of rows of A, needs to be square!
  const int nnzA  = 7; // number of nonzeros of A

  std::vector<T> coeffs;

  coeffs.push_back(T(0,0,1.0));
  coeffs.push_back(T(0,1,4.0));
  coeffs.push_back(T(1,1,2.0));
  coeffs.push_back(T(1,2,3.0));
  coeffs.push_back(T(2,0,5.0));
  coeffs.push_back(T(2,3,7.0));
  coeffs.push_back(T(3,2,9.0));

  SpMatRowMaj A(rowsA,colsA);
  A.setFromTriplets(coeffs.begin(),coeffs.end());

  A.makeCompressed();

  // Set up RHS
  Eigen::VectorXd b(rowsA); // Assuming real == double
  b << 1, 2, 3, 4; 

  real *h_csrValA = A.valuePtr();
  real *h_b = b.data();
  int *h_csrColIndA = A.innerIndexPtr();
  int *h_csrRowPtrA = A.outerIndexPtr();

  // Set up CUDA
  cusolverSpHandle_t handle = NULL;
  cusparseMatDescr_t descrA = NULL;

  checkCudaErrors(cusolverSpCreate(&handle));
  checkCudaErrors(cusparseCreateMatDescr(&descrA));
  checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
  
  int *d_csrRowPtrA = NULL;
  int *d_csrColIndA = NULL;
  double *d_csrValA = NULL;
  double *d_x = NULL; // x = A \ b
  double *d_b = NULL; // a copy of h_b
  
  double tol = 1.e-12;
  int reorder = 0; // no reordering
  int singularity = 0; // -1 if A is invertible under tol.

  // Prepare data on device
  checkCudaErrors(cudaMalloc((void **)&d_csrRowPtrA, sizeof(int)*(rowsA+1)));
  checkCudaErrors(cudaMalloc((void **)&d_csrColIndA, sizeof(int)*nnzA));
  checkCudaErrors(cudaMalloc((void **)&d_csrValA   , sizeof(double)*nnzA));
  checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(double)*colsA));
  checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(double)*rowsA));

  checkCudaErrors(
      cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, sizeof(int)*(rowsA+1), cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int)*nnzA     , cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_csrValA   , h_csrValA   , sizeof(double)*nnzA  , cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_b, h_b, sizeof(double)*rowsA, cudaMemcpyHostToDevice));
 
  // Solve the system  
  checkCudaErrors(cusolverSpDcsrlsvqr(
      handle, rowsA, nnzA,
      descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
      d_b, tol, reorder, d_x, &singularity));
 
  std::ostringstream sing_err;
  sing_err << "WARNING: the matrix is singular at row " << singularity << "  under tol (" << tol << ")";
  BOOST_REQUIRE_MESSAGE(0 > singularity, sing_err.str());

  // Copy results back into b
  checkCudaErrors(cudaMemcpy(b.data(), d_x, sizeof(double)*rowsA, cudaMemcpyDeviceToHost));

  BOOST_CHECK_CLOSE(b[0],-1./3.,1e-10);
  BOOST_CHECK_CLOSE(b[1], 1./3.,1e-10);
  BOOST_CHECK_CLOSE(b[2], 4./9.,1e-10);
  BOOST_CHECK_CLOSE(b[3], 2./3.,1e-10);


  if (handle) { checkCudaErrors(cusolverSpDestroy(handle)); }
  if (descrA) { checkCudaErrors(cusparseDestroyMatDescr(descrA)); }
  if (d_csrValA   ) { checkCudaErrors(cudaFree(d_csrValA)); }
  if (d_csrRowPtrA) { checkCudaErrors(cudaFree(d_csrRowPtrA)); }
  if (d_csrColIndA) { checkCudaErrors(cudaFree(d_csrColIndA)); }
  if (d_x) { checkCudaErrors(cudaFree(d_x)); }
  if (d_b) { checkCudaErrors(cudaFree(d_b)); }
}

BOOST_AUTO_TEST_CASE(QRDense)
{
  // Try a dense matrix. This may be inefficient but we want to be sure
  // this works
 
  const int colsA = 2; // number of columns of A
  const int rowsA = colsA; // number of rows of A, needs to be square!

  std::vector<T> coeffs;

  coeffs.push_back(T(0,0,1.2));
  coeffs.push_back(T(0,1,0.8));
  coeffs.push_back(T(1,0,1.1));
  coeffs.push_back(T(1,1,1.3));

  SpMatRowMaj A(rowsA,colsA);
  A.setFromTriplets(coeffs.begin(),coeffs.end());

  A.makeCompressed();

  const int nnzA  = A.nonZeros(); // number of nonzeros of A

  // Set up RHS
  Eigen::VectorXd b(rowsA); // Assuming real == double
  b << 200.2, 239.7;

  real *h_csrValA = A.valuePtr();
  real *h_b = b.data();
  int *h_csrColIndA = A.innerIndexPtr();
  int *h_csrRowPtrA = A.outerIndexPtr();

  // Set up CUDA
  cusolverSpHandle_t handle = NULL;
  cusparseMatDescr_t descrA = NULL;

  checkCudaErrors(cusolverSpCreate(&handle));
  checkCudaErrors(cusparseCreateMatDescr(&descrA));
  checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
  
  int *d_csrRowPtrA = NULL;
  int *d_csrColIndA = NULL;
  double *d_csrValA = NULL;
  double *d_x = NULL; // x = A \ b
  double *d_b = NULL; // a copy of h_b
  
  double tol = 1.e-12;
  int reorder = 0; // no reordering
  int singularity = 0; // -1 if A is invertible under tol.

  // Prepare data on device
  checkCudaErrors(cudaMalloc((void **)&d_csrRowPtrA, sizeof(int)*(rowsA+1)));
  checkCudaErrors(cudaMalloc((void **)&d_csrColIndA, sizeof(int)*nnzA));
  checkCudaErrors(cudaMalloc((void **)&d_csrValA   , sizeof(double)*nnzA));
  checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(double)*colsA));
  checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(double)*rowsA));

  checkCudaErrors(
      cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, sizeof(int)*(rowsA+1), cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int)*nnzA     , cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_csrValA   , h_csrValA   , sizeof(double)*nnzA  , cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_b, h_b, sizeof(double)*rowsA, cudaMemcpyHostToDevice));
 
  // Solve the system  
  checkCudaErrors(cusolverSpDcsrlsvqr(
      handle, rowsA, nnzA,
      descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
      d_b, tol, reorder, d_x, &singularity));
 
  std::ostringstream sing_err;
  sing_err << "WARNING: the matrix is singular at row " << singularity << "  under tol (" << tol << ")";
  BOOST_REQUIRE_MESSAGE(0 > singularity, sing_err.str());

  // Copy results back into b
  checkCudaErrors(cudaMemcpy(b.data(), d_x, sizeof(double)*rowsA, cudaMemcpyDeviceToHost));

  BOOST_CHECK_CLOSE(b[0], 100.73529411764707,1e-10);
  BOOST_CHECK_CLOSE(b[1],  99.14705882352939,1e-10);


  if (handle) { checkCudaErrors(cusolverSpDestroy(handle)); }
  if (descrA) { checkCudaErrors(cusparseDestroyMatDescr(descrA)); }
  if (d_csrValA   ) { checkCudaErrors(cudaFree(d_csrValA)); }
  if (d_csrRowPtrA) { checkCudaErrors(cudaFree(d_csrRowPtrA)); }
  if (d_csrColIndA) { checkCudaErrors(cudaFree(d_csrColIndA)); }
  if (d_x) { checkCudaErrors(cudaFree(d_x)); }
  if (d_b) { checkCudaErrors(cudaFree(d_b)); }
}

BOOST_AUTO_TEST_SUITE_END()

#endif
