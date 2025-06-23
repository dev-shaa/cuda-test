#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>

#define N 32

__global__ void foo(float *values)
{
    values[threadIdx.x] = threadIdx.x;
}

int main(int argc, char const *argv[])
{
    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    assert(gpu_count > 0);

    int count = N * gpu_count;
    float *host_values;
    cudaMallocHost(&host_values, count * sizeof(float));

    cudaStream_t streams[8];
    for (int i = 0; i < gpu_count; i++)
    {
        float *dev_values;

        cudaSetDevice(i);
        cudaStreamCreate(&(streams[i]));
        cudaMallocAsync(&dev_values, N * sizeof(float), streams[i]);
        cudaMemcpyAsync(dev_values, host_values + i * N, N * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

        foo<<<1, N>>>(dev_values);

        cudaMemcpyAsync(host_values + i * N, dev_values, N * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
        cudaFreeAsync(dev_values, streams[i]);
    }

    for (int i = 0; i < gpu_count; i++)
    {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    for (int i = 0; i < count; i++)
        printf("%.f ", host_values[i]);
    printf("\n");

    cudaFreeHost(host_values);

    return 0;
}
