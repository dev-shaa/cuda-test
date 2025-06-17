#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublasXt.h>
#include <assert.h>

#define M 8
#define K M
#define N M

void phpc_gemm_cublas(const double *a, int lda, const double *b, int ldb, double *c, int ldc, int m, int k, int n, int gpu_count)
{
    assert(gpu_count < 32);

    int devices[32];
    cublasXtHandle_t handle;
    double alpha = 1, beta = 1;

    for (size_t i = 0; i < gpu_count; i++)
        devices[i] = i;

    cublasXtCreate(&handle);
    cublasXtDeviceSelect(handle, gpu_count, devices);

    cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, ldb, a, lda, &beta, c, ldc);
    cublasXtDestroy(handle);
}

int main(int argc, char const *argv[])
{
#define LDA (K + 8)
#define LDB (N + 4)
#define LDC (N + 3)

    int gpu_count;
    cudaGetDeviceCount(&gpu_count);

    printf("GPUS: %d\n", gpu_count);

    double a[M * LDA];
    double b[K * LDB];
    double c[M * LDC];

    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < K; j++)
            a[i * LDA + j] = i + 1;
    }

    for (size_t i = 0; i < K; i++)
    {
        for (size_t j = 0; j < N; j++)
            b[i * LDB + j] = j + 1;
    }

    for (size_t i = 0; i < M * LDC; i++)
        c[i] = 0;

    printf("A:\n");
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < LDA; j++)
            printf("%.lf ", a[i * LDA + j]);

        printf("\n");
    }

    printf("B:\n");
    for (size_t i = 0; i < K; i++)
    {
        for (size_t j = 0; j < LDB; j++)
            printf("%.lf ", b[i * LDB + j]);

        printf("\n");
    }

    phpc_gemm_cublas(a, LDA, b, LDB, c, LDC, M, K, N, gpu_count);

    printf("C:\n");
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < LDC; j++)
            printf("%.lf ", c[i * LDC + j]);

        printf("\n");
    }

#undef LDC
#undef LDB
#undef LDA

    return 0;
}
