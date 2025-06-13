#ifndef UTILS_CU
#define UTILS_CU

__global__ void foo(int *values)
{
    values[threadIdx.x] += 1;
}

#endif
