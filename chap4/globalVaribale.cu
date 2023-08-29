#include <cuda_runtime.h>
#include <stdio.h>

__device__ float devA;

__global__ void addDevA() {
    devA += 10;
}

int main() {
    float a = 2.333;
    printf("Value %f\n", a);
    cudaMemcpyToSymbol(devA, &a, sizeof(a));
    addDevA<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&a, devA, sizeof(a));
    printf("Value %f\n", a);

    cudaDeviceReset();
    return 0;
}