#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>

__global__ void foo(float *values)
{
    uint id = blockDim.x * blockIdx.x + threadIdx.x;
    values[id] = sin(values[id]);
}

int main(int argc, char const *argv[])
{
    assert(argc > 1);
    int count = atoi(argv[1]);
    assert(count > 0 && count % 32 == 0);

    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    assert(gpu_count > 0);

    int N = gpu_count * count;
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
        cudaMallocAsync(&dev_values, count * sizeof(float), streams[i]);
        cudaMemcpyAsync(dev_values, host_values + i * count, count * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

        foo<<<count / 32, 32>>>(dev_values);

        cudaMemcpyAsync(host_values + i * count, dev_values, count * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
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
