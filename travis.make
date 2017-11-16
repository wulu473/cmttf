
# Travis CI

# Not yet working

CUDA_PATH=/usr/local/cuda-8.0

CXX=g++
NVCC=nvcc -ccbin $(CXX)

INCLUDES=-isystem=/usr/local/include/eigen3 -isystem=$(CUDA_PATH)/samples/common/inc

# Everything should be in the default
LDFLAGS=

-include common.make

