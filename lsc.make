
# LSC - Ubuntu 16.04 machines

CUDA_PATH=/lsc/opt/cuda-8.0

CXX=g++-6 
NVCC=$(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

INCLUDES=-isystem=/lsc/opt/modules/gcc-6.4.0/boost-1.61.0/include -isystem=/home/raid/lw473/opt/eigen-3.2.8/include/eigen3 -isystem=$(CUDA_PATH)/samples/common/inc  -isystem=/lsc/opt/modules/gcc-6.4.0/libconfig-1.5/include -isystem=/home/raid/lw473/opt/expertk-2.7/

LDFLAGS=-L/lsc/opt/modules/gcc-6.4.0/boost-1.61.0/lib -L$(CUDA_PATH)/lib64/ -L/lsc/opt/modules/gcc-6.4.0/libconfig-1.5/lib

-include common.make
