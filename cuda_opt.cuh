#ifndef _CUDA_OPT_
#define _CUDA_OPT_
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include "./eigen-3.4.0/Eigen/Dense"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


void concatXBiasKernelUse(float *A, float *B, float *C, int Crows, int Ccols);
void matMulKernelUse(float *A, float *B, float *C, int Arows, int Acols, int Bcols);
void sigmoidKernelUse(float *A, float *B, int Brows, int Bcols);
void matSubKernelUse(float *A, float *B, float *C, int Crows, int Ccols);
void matErrorKernelUse(float *A, float *B, int Brows, int Bcols, float batch);
void matMulBTKernelUse(float *A, float *B, float *C, int Arows, int Acols, int Brows);
void matMulATKernelUse(float *A, float *B, float *C, int Arows, int Acols, int Bcols);
void sigmoidDiffKenelUse(float *A, float *B, float *C, int Arows, int Acols);
void copyNoBiasKernelUse(float *A, float *B, int Brows, int Bcols);
void weightUpdateKernelUse(float *A, float *B, int Brows, int Bcols, float lr);
__global__ void matErrorKernel(float *A, float *B, int Brows, int Bcols, float batch);
__global__ void matSubKernel(float *A, float *B, float *C, int Crows, int Ccols);
__device__ float sigmoid(float x);
__device__ float sigmoid_diff(float x);
__global__ void copyNoBiasKernel(float *A, float *B, int Brows, int Bcols);
__global__ void sigmoidDiffKenel(float *A, float *B, float *C, int Arows, int Acols);
__global__ void sigmoidKernel(float *A, float *B, int Brows, int Bcols);
__global__ void concatXBiasKernel(float *A, float *B, float *C, int Crows, int Ccols);
__global__ void weightUpdateKernel(float *A, float *B, int Brows, int Bcols, float lr);
__global__ void matMulBTKernel(float *A, float *B, float *C, int Arows, int Acols, int Brows);
__global__ void matMulATKernel(float *A, float *B, float *C, int Arows, int Acols, int Bcols);
__global__ void matMulKernel(float *A, float *B, float *C, int Arows, int Acols, int Bcols);

void cudaMatMul(const Eigen::MatrixXf &A, const Eigen::MatrixXf &B, Eigen::MatrixXf &C);



#endif






