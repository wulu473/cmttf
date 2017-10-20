
#ifndef CUDA_H_
#define CUDA_H_

#include <map>
#include <cuda_runtime.h>

class CUDA
{
  public:
    //! Return the device ID
    static unsigned int device();

    // Set the device ID
    static void setDevice(const unsigned int);

    static unsigned int numDevices();

    static std::map<unsigned int, unsigned int> freeMemory();

    static std::map<unsigned int, cudaDeviceProp> deviceProperties();

    static unsigned int deviceMostFreeMemory();

  private:
    static unsigned int m_device;
};

#endif
