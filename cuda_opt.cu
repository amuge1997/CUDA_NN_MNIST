
#include "cuda_opt.cuh"

__device__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}
__device__ float sigmoid_diff(float x)
{
    return x * (1.0f - x);
}

__global__ void matErrorKernel(float *A, float *B, int Brows, int Bcols, float batch)
{
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    if (yi < Brows && xi < Bcols)
    {
        int index = Bcols * yi + xi;
        B[index] = -2.f / batch * A[index];
    }
}

void matErrorKernelUse(float *A, float *B, int Brows, int Bcols, float batch)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((Bcols + threadsPerBlock.x - 1) / threadsPerBlock.x, (Brows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matErrorKernel<<<numBlocks, threadsPerBlock>>>(A, B, Brows, Bcols, batch);
}

__global__ void matSubKernel(float *A, float *B, float *C, int Crows, int Ccols)
{
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    if (yi < Crows && xi < Ccols)
    {
        int index = Ccols * yi + xi;
        C[index] = A[index] - B[index];
    }
}

void matSubKernelUse(float *A, float *B, float *C, int Crows, int Ccols)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((Ccols + threadsPerBlock.x - 1) / threadsPerBlock.x, (Crows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matSubKernel<<<numBlocks, threadsPerBlock>>>(A, B, C, Crows, Ccols);
}

__global__ void copyNoBiasKernel(float *A, float *B, int Brows, int Bcols) // 默认 B 比 A 少一行
{
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    if (yi < Brows && xi < Bcols)
    {
        int index = Bcols * yi + xi;
        B[index] = A[index];
    }
}

void copyNoBiasKernelUse(float *A, float *B, int Brows, int Bcols)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((Bcols + threadsPerBlock.x - 1) / threadsPerBlock.x, (Brows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    copyNoBiasKernel<<<numBlocks, threadsPerBlock>>>(A, B, Brows, Bcols);
}

__global__ void sigmoidDiffKenel(float *A, float *B, float *C, int Arows, int Acols)
{
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    if (yi < Arows && xi < Acols)
    {
        int index = Acols * yi + xi;
        C[index] = A[index] * sigmoid_diff(B[index]);
    }
}

void sigmoidDiffKenelUse(float *A, float *B, float *C, int Arows, int Acols)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((Acols + threadsPerBlock.x - 1) / threadsPerBlock.x, (Arows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    sigmoidDiffKenel<<<numBlocks, threadsPerBlock>>>(A, B, C, Arows, Acols);
}

__global__ void sigmoidKernel(float *A, float *B, int Brows, int Bcols)
{
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    if (yi < Brows && xi < Bcols)
    {
        int index = Bcols * yi + xi;
        B[index] = sigmoid(A[index]);
    }
}

void sigmoidKernelUse(float *A, float *B, int Brows, int Bcols)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((Bcols + threadsPerBlock.x - 1) / threadsPerBlock.x, (Brows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    sigmoidKernel<<<numBlocks, threadsPerBlock>>>(A, B, Brows, Bcols);
}

__global__ void concatXBiasKernel(float *A, float *B, float *C, int Crows, int Ccols)
{
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    if (yi < Crows && xi < Ccols)
    {
        if (yi < Crows - 1)
        {
            int index = Ccols * yi + xi;
            C[index] = A[index];
        }
        else
        {
            C[Ccols * yi + xi] = B[xi];
        }
    }
}

void concatXBiasKernelUse(float *A, float *B, float *C, int Crows, int Ccols)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((Ccols + threadsPerBlock.x - 1) / threadsPerBlock.x, (Crows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    concatXBiasKernel<<<numBlocks, threadsPerBlock>>>(A, B, C, Crows, Ccols);
}

__global__ void weightUpdateKernel(float *A, float *B, int Brows, int Bcols, float lr)
{
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    if (yi < Brows && xi < Bcols)
    {
        int index = Bcols * yi + xi;
        B[index] = B[index] - lr * A[index];
    }
}

void weightUpdateKernelUse(float *A, float *B, int Brows, int Bcols, float lr)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((Bcols + threadsPerBlock.x - 1) / threadsPerBlock.x, (Brows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    weightUpdateKernel<<<numBlocks, threadsPerBlock>>>(A, B, Brows, Bcols, lr);
}

__global__ void matMulBTKernel(float *A, float *B, float *C, int Arows, int Acols, int Brows)
{
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    if (yi < Arows && xi < Brows)
    {
        float Cvalue = 0;
        for (int e = 0; e < Acols; ++e)
        {
            int Aindex = yi * Acols + e;
            int Bindex = xi * Acols + e;
            Cvalue += A[Aindex] * B[Bindex];
        }
        C[yi * Brows + xi] = Cvalue;
    }
}

void matMulBTKernelUse(float *A, float *B, float *C, int Arows, int Acols, int Brows)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((Brows + threadsPerBlock.x - 1) / threadsPerBlock.x, (Arows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matMulBTKernel<<<numBlocks, threadsPerBlock>>>(A, B, C, Arows, Acols, Brows);
}

__global__ void matMulATKernel(float *A, float *B, float *C, int Arows, int Acols, int Bcols)
{
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    if (yi < Acols && xi < Bcols)
    {
        float Cvalue = 0;
        for (int e = 0; e < Arows; ++e)
        {
            int Aindex = e * Acols + yi;
            int Bindex = e * Bcols + xi;
            Cvalue += A[Aindex] * B[Bindex];
            // printf("%d-%d %d-%d %.2f-%.2f\n", yi, xi, Aindex, Bindex, A[Aindex], B[Bindex]);
        }
        C[yi * Bcols + xi] = Cvalue;
        // printf("%d %.2f\n", yi * Bcols + xi, C[yi * Bcols + xi]);
    }
}

void matMulATKernelUse(float *A, float *B, float *C, int Arows, int Acols, int Bcols)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((Bcols + threadsPerBlock.x - 1) / threadsPerBlock.x, (Acols + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matMulATKernel<<<numBlocks, threadsPerBlock>>>(A, B, C, Arows, Acols, Bcols);
}

__global__ void matMulKernel(float *A, float *B, float *C, int Arows, int Acols, int Bcols)
{
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    if (yi < Arows && xi < Bcols)
    {
        float Cvalue = 0;
        for (int e = 0; e < Acols; ++e)
        {
            int Aindex = yi * Acols + e;
            int Bindex = e * Bcols + xi;
            Cvalue += A[Aindex] * B[Bindex];
        }
        C[yi * Bcols + xi] = Cvalue;
    }
}

void matMulKernelUse(float *A, float *B, float *C, int Arows, int Acols, int Bcols)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((Bcols + threadsPerBlock.x - 1) / threadsPerBlock.x, (Arows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matMulKernel<<<numBlocks, threadsPerBlock>>>(A, B, C, Arows, Acols, Bcols);
}

void cudaMatMul(const Eigen::MatrixXf &A, const Eigen::MatrixXf &B, Eigen::MatrixXf &C)
{
    float *d_A, *d_B, *d_C;
    Eigen::MatrixXf At = A.transpose();
    Eigen::MatrixXf Bt = B.transpose();
    cudaMalloc((void **)&d_A, A.rows() * A.cols() * sizeof(float));
    cudaMalloc((void **)&d_B, B.rows() * B.cols() * sizeof(float));
    cudaMalloc((void **)&d_C, A.rows() * B.cols() * sizeof(float));
    cudaMemcpy(d_A, At.data(), A.rows() * A.cols() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, Bt.data(), B.rows() * B.cols() * sizeof(float), cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((B.cols() + threadsPerBlock.x - 1) / threadsPerBlock.x, (A.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, A.rows(), A.cols(), B.cols());
    C.resize(B.cols(), A.rows()); // 由于是列优先存储，因此先转置，再写入，最后再转置回来
    cudaMemcpy(C.data(), d_C, A.rows() * B.cols() * sizeof(float), cudaMemcpyDeviceToHost);
    C.transposeInPlace();
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
