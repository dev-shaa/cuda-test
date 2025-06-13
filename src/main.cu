#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 32

__global__ void foo(int *values)
{
    values[threadIdx.x] += 1;
}

int main(int argc, char const *argv[])
{
    int *host_values = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
        host_values[i] = i;

    int *dev_values;
    cudaMalloc(&dev_values, N * sizeof(int));
    cudaMemcpy(dev_values, host_values, N * sizeof(int), cudaMemcpyHostToDevice);

    foo<<<1, N>>>(dev_values);

    cudaDeviceSynchronize();
    cudaMemcpy(host_values, dev_values, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        printf("%d ", host_values[i]);
    printf("\n");

    cudaFree(dev_values);
    free(host_values);

    return 0;
}
