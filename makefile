CUDA_INSTALL_PATH := /usr/local/cuda
CUDA_LIB       := -L $(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart

all: NAME GPU MAIN CLEAN

NAME:
        cp *.cpp gpu_x37.cu

GPU:
        nvcc -O3 -c *.cu

MAIN_OpenMp:
        g++ *.o $(CUDA_LIB) -fopenmp -O3 -o test.run

MAIN:
        g++ *.o $(CUDA_LIB) -O3 -o test.run

CLEAN:
        rm *.o *.cu

EXE:
        ./test.run
