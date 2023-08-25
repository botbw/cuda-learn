#include "common.h"
#include <time.h>
#include <stdlib.h>

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match)
        printf("Arrays match.\n\n");
}

void initialData(float *a, int sz)
{
    srand(time(NULL));
    for (int i = 0; i < sz; i++)
    {
        a[i] = rand() / 1.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
        C[idx] = A[idx] + B[idx];
}

__global__ void sumArraysOnDevice(float *a, float *b, float *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    int dev = 0;
    cudaSetDevice(dev);

    unsigned nElem = 32;

    float *a, *b, *c_host, *c_dev;
    size_t memSz = nElem * sizeof(float);
    a = (float*)malloc(memSz);
    b = (float*)malloc(memSz);
    c_host = (float*)malloc(memSz);
    c_dev = (float*)malloc(memSz);

    initialData(a, nElem);
    initialData(b, nElem);

    float *da, *db, *dc;
    cudaMalloc(&da, memSz);
    cudaMalloc(&db, memSz);
    cudaMalloc(&dc, memSz);

    cudaMemcpy(da, a, memSz, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, memSz, cudaMemcpyHostToDevice);

    dim3 block{nElem};
    dim3 grid{nElem / block.x};

    printf("Execution configuration: <<<%d, %d>>>\n", grid.x, block.x);

    sumArraysOnDevice<<<grid, block>>>(da, db, dc);
    
    cudaMemcpy(c_dev, dc, memSz, cudaMemcpyDeviceToHost);
    
    sumArraysOnHost(a, b, c_host, nElem);

    checkResult(c_host, c_dev, nElem);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    free(a);
    free(b);
    free(c_host);
    free(c_dev);

    return 0;
}