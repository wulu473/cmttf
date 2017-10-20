
#include "CUDA.hpp"
#include "Log.hpp"

#include <list>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

unsigned int CUDA::m_device = 0;

unsigned int CUDA::device()
{
  return m_device;
}

unsigned int CUDA::numDevices()
{
  int nDev;
  cudaGetDeviceCount(&nDev);
  return (unsigned int) nDev;
}

// Return a map of device ID to device properties
std::map<unsigned int, cudaDeviceProp> CUDA::deviceProperties()
{
  std::map<unsigned int, cudaDeviceProp> props;
  for(unsigned int id=0; id<numDevices(); id++)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, id);
    props.insert(std::pair<unsigned int, cudaDeviceProp>(id, prop));
  }
  return props;
}

// Return a map of device ID to free memory
std::map<unsigned int, unsigned int> CUDA::freeMemory()
{
  const unsigned int oldDev = device();

  std::map<unsigned int, unsigned int> freeMem;
  for(unsigned int id=0;id<numDevices();id++)
  {
    setDevice(id);
    size_t total, free;
    cuMemGetInfo(&free, &total);
    freeMem.insert(std::pair<unsigned int, unsigned int>(id, free));
  } 
  setDevice(oldDev);
  return freeMem;
}

unsigned int CUDA::deviceMostFreeMemory()
{
  typedef std::pair<unsigned int, unsigned int> DevMemPr;
  const auto devFreeMemMap = freeMemory();
  std::list<DevMemPr> devFreeMem(
      devFreeMemMap.begin(), devFreeMemMap.end());
  devFreeMem.sort([](const DevMemPr &left, const DevMemPr& right){ return (bool)(left.second < right.second); } );
  return devFreeMem.back().first;
}

void CUDA::setDevice(const unsigned int device)
{
  checkCudaErrors(cudaSetDevice(device));
  cudaFree(0); // establishes cuda context
  m_device = device;
}

