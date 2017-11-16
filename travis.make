
# Travis CI

# Not yet working

CUDA_PATH=/lsc/opt/cuda-8.0

CXX=g++
NVCC=$(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

INCLUDES=-isystem=/usr/local/include/eigen3 -isystem=/home/raid/lw473/opt/expertk-2.7/

# Everything should be in the default
LDFLAGS=

-include common.make

