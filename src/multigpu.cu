#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "utils.cu"

#define N 32

int main(int argc, char const *argv[])
{
    int *host_values = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
        host_values[i] = i;

    int gpu_count;
    cudaGetDeviceCount(&gpu_count);

    int size = N / gpu_count;
    cudaStream_t *streams = (cudaStream_t *)malloc(gpu_count * sizeof(cudaStream_t));

    for (int i = 0; i < gpu_count; i++)
    {
        cudaSetDevice(i);
        cudaStreamCreate(&(streams[i]));

        int *dev_values;
        cudaMallocAsync(&dev_values, size * sizeof(int), streams[i]);
        cudaMemcpyAsync(dev_values, host_values + size * i, size * sizeof(int), cudaMemcpyHostToDevice, streams[i]);

        foo<<<1, size, 0, streams[i]>>>(dev_values);

        cudaMemcpyAsync(host_values + size * i, dev_values, size * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
        cudaFreeAsync(dev_values, streams[i]);
    }

    for (int i = 0; i < gpu_count; i++)
    {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    for (int i = 0; i < N; i++)
        printf("%d ", host_values[i]);
    printf("\n");

    free(streams);
    free(host_values);

    return 0;
}
