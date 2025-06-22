#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>

#define COUNT 32

__global__ void foo(float *values)
{
    values[threadIdx.x] += 1;
}

int main(int argc, char const *argv[])
{
    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    assert(gpu_count > 0);

    int N = gpu_count * COUNT;
    float *host_values;
    cudaMallocHost(&host_values, N * sizeof(float));
    for (int i = 0; i < N; i++)
        host_values[i] = i;

    cudaStream_t streams[8];
    for (int i = 0; i < gpu_count; i++)
    {
        float *dev_values;

        cudaSetDevice(i);
        cudaStreamCreate(&(streams[i]));
        cudaMallocAsync(&dev_values, COUNT * sizeof(float), streams[i]);
        cudaMemcpyAsync(dev_values, host_values + i * COUNT, COUNT * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

        foo<<<1, COUNT>>>(dev_values);

        cudaMemcpyAsync(host_values + i * COUNT, dev_values, COUNT * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
        cudaFreeAsync(dev_values, streams[i]);
    }

    for (int i = 0; i < gpu_count; i++)
    {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    for (int i = 0; i < N; i++)
        printf("%.f ", host_values[i]);

    printf("\n");

    cudaFreeHost(host_values);

    return 0;
}
