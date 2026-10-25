#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#define TILE 16

__global__ void matmul(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < K; k++)
        {
            sum += A[row * K + k] * B[N * k + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void tiled_matmul(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K)
{
    __shared__ float tile_A[TILE][TILE];
    __shared__ float tile_B[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++)
    {

        // ① A의 입력 타일 적재
        int a_col = t * TILE + tx;

        if (row < M && a_col < K)
            tile_A[ty][tx] = A[row * K + a_col];
        else
            tile_A[ty][tx] = 0.0f;

        // ② B의 입력 타일 적재
        int b_row = t * TILE + ty;

        if (b_row < K && col < N)
            tile_B[ty][tx] = B[b_row * N + col];
        else
            tile_B[ty][tx] = 0.0f;

        // ③ 모든 스레드가 데이터 적재를 마칠 때까지 대기
        __syncthreads();

        // ④ 각 스레드가 자신의 출력 원소에 대한 부분합 계산
        for (int k = 0; k < TILE; k++)
        {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }

        // ⑤ 다음 입력 타일로 덮어쓰기 전에 대기
        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
    // TODO: 최종 결과 저장
}

int main()
{

    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_naive(M * N);
    std::vector<float> h_C_tiled(M * N);

    for (int i = 0; i < M * K; i++)
        h_A[i] = static_cast<float>(i % 7);

    for (int i = 0; i < K * N; i++)
        h_B[i] = static_cast<float>(i % 5);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C_naive;
    float *d_C_tiled;

    cudaMalloc((void **)&d_A, sizeof(float) * M * K);
    cudaMalloc((void **)&d_B, sizeof(float) * K * N);
    cudaMalloc((void **)&d_C_naive, M * N * sizeof(float));
    cudaMalloc((void **)&d_C_tiled, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks(
        (N + 15) / 16,
        (M + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int WARMUP = 10;
    const int REPEAT = 100;

    // Naive MatMul
    for (int i = 0; i < WARMUP; i++)
        matmul<<<blocks, threads>>>(d_A, d_B, d_C_naive, M, N, K);

    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for (int i = 0; i < REPEAT; i++)
        matmul<<<blocks, threads>>>(d_A, d_B, d_C_naive, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float naive_ms = 0.0f;
    cudaEventElapsedTime(&naive_ms, start, stop);
    naive_ms /= REPEAT;

    // Tiled MatMul
    for (int i = 0; i < WARMUP; i++)
        tiled_matmul<<<blocks, threads>>>(d_A, d_B, d_C_tiled, M, N, K);

    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for (int i = 0; i < REPEAT; i++)
        tiled_matmul<<<blocks, threads>>>(d_A, d_B, d_C_tiled, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float tiled_ms = 0.0f;
    cudaEventElapsedTime(&tiled_ms, start, stop);
    tiled_ms /= REPEAT;

    std::cout << "Naive:   " << naive_ms << " ms\n";
    std::cout << "Tiled:   " << tiled_ms << " ms\n";
    std::cout << "Speedup: " << naive_ms / tiled_ms << "x\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(err) << '\n';
    }
    float max_error = 0.0f;

    for (int i = 0; i < M * N; i++)
    {
        float error = fabsf(h_C_tiled[i] - h_C_naive[i]);
        max_error = std::max(max_error, error);
    }

    std::cout << "Max error: " << max_error << '\n';
}