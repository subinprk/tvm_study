#include <iostream>
#include <cuda_runtime.h>

__global__ void matmul(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < K; k ++)
        {
            sum += A[row * K + k] * B[N * k + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    const int M = 2;
    const int N = 2;
    const int K = 3;

    float h_A[M * K] = {1, 2, 3, 4, 5, 6};
    float h_B[K * N] = {7, 8, 9, 10, 11, 12};
    float h_C[M * N] = {};

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    cudaMalloc((void**)& d_A, sizeof(float) * M * K);
    cudaMalloc((void**)& d_B, sizeof(float) * K * N);
    cudaMalloc((void**)& d_C, sizeof(float) * M * N);

    cudaMemcpy(d_A, h_A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks(
        (N + 15) / 16,
        (M + 15) / 16
    );

    matmul <<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess){
        std::cerr << cudaGetErrorString(err) << '\n';
    }

    cudaMemcpy(h_C, d_C, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; i ++){
        for (int j = 0; j < N; j++){
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << '\n';
    }
}