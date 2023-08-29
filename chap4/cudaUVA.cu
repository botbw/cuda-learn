#include <cuda_runtime.h>
#include <stdio.h>

__global__ void sum(int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if(tid < n) c[tid] = a[tid] + b[tid];
}

int main() {

    int n = 10;
    int *a = NULL, *b = NULL, *c = NULL;

    cudaHostAlloc(&a, n * sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc(&b, n * sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc(&c, n * sizeof(int), cudaHostAllocMapped);

    for(int i = 0 ; i < n; i++) {
        a[i] = b[i] = i;
    }

    sum<<<1, n>>>(a, b, c, n);
    cudaDeviceSynchronize();
    for(int i = 0; i < n; i++) {
        printf("%d ", c[i]);
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    cudaDeviceReset();
    return 0;
}