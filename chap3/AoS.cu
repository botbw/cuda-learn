#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call) \
{ \
 const cudaError_t error = call; \
 if (error != cudaSuccess) \
 { \
 printf("Error: %s:%d, ", __FILE__, __LINE__); \
 printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
 exit(1); \
 } \
}

#include <sys/time.h>
inline double seconds()
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);
}

struct tmp
{
    float a, b;
};

void initialData(float *ip, int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 1.0f;
    }
}

void sumArrays(float *a, float *b, float *c, int sz)
{
    for (int i = 0; i < sz; i++)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void sumArraysGPU(float *a, float *b, struct tmp *c, int sz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
    {
        c[i].a = a[i] + b[i];
    }
}

void checkResult_struct(float *res_h, struct tmp *res_from_gpu_h, int nElem)
{
    for (int i = 0; i < nElem; i++)
        if (res_h[i] != res_from_gpu_h[i].a)
        {
            printf("check fail!\n");
            exit(0);
        }
    printf("result check success!\n");
}

int main(int argc, char **argv)
{
    int dev = 0;
    cudaSetDevice(dev);
    int nElem = 1 << 18;
    int offset = 0;
    if (argc >= 2)
        offset = atoi(argv[1]);
    printf("Vector size:%d\n", nElem);
    int nByte = sizeof(float) * nElem;
    int nByte_struct = sizeof(struct tmp) * nElem;
    float *a_h = (float *)malloc(nByte);
    float *b_h = (float *)malloc(nByte);
    float *res_h = (float *)malloc(nByte_struct);
    struct tmp *res_from_gpu_h = (struct tmp *)malloc(nByte_struct);
    memset(res_h, 0, nByte);
    memset(res_from_gpu_h, 0, nByte);
    float *a_d, *b_d;
    struct tmp *res_d;
    CHECK(cudaMalloc((void **)&a_d, nByte));
    CHECK(cudaMalloc((void **)&b_d, nByte));
    CHECK(cudaMalloc((void **)&res_d, nByte_struct));
    CHECK(cudaMemset(res_d, 0, nByte_struct));
    initialData(a_h, nElem);
    initialData(b_h, nElem);

    CHECK(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));

    dim3 block(1024);
    dim3 grid(nElem / block.x);
    double iStart, iElaps;
    iStart = seconds();
    sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d, nElem);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte_struct, cudaMemcpyDeviceToHost));
    printf("Execution configuration<<<%d,%d>>> Time elapsed %f sec\n", grid.x, block.x, iElaps);

    sumArrays(a_h, b_h, res_h, nElem);

    checkResult_struct(res_h, res_from_gpu_h, nElem);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);

    free(a_h);
    free(b_h);
    free(res_h);
    free(res_from_gpu_h);

    return 0;
}