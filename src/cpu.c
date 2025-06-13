#include <stdio.h>
#include <stdlib.h>

#define N 1024

int main(int argc, char const *argv[])
{
    int host_values[N];
    for (int i = 0; i < N; i++)
        host_values[i] = i;

    for (int i = 0; i < N; i++)
        host_values[i] += 1;

    for (int i = 0; i < N; i++)
        printf("%d ", host_values[i]);
    printf("\n");

    return 0;
}
