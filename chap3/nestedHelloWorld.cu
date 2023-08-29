#include <cuda_runtime.h>
#include <stdio.h>

typedef unsigned int uint;

__global__ void nestedHelloWorld(int thNum, int dep)
{
    uint tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d"
           " block %d\n",
           dep, tid, blockIdx.x);
    if (thNum <= 1)
        return;

    if (tid == 0)
    {
        nestedHelloWorld<<<1, thNum>>>(thNum / 2, dep + 1);
        printf("-------> nested execution depth: %d\n", dep);
    }
}

int main()
{
    nestedHelloWorld<<<1, 8>>>(4, 0);
    cudaDeviceSynchronize();
}