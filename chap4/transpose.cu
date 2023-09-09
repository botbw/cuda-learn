#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>
double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

void initialData(float *ip, const int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = 1.0f;
    }
}

void transposeHost(float *out, float *in, int x, int y)
{
    for(int i = 0; i < x; i++) {
        for(int j = 0; j < y; j++) {
            out[j * x + i] = in[i * y + j];
        }
    }
}

__global__ void warmup(float *out, float *in, int x, int y)
{
    int thdX = blockIdx.x * blockDim.x + threadIdx.x;
    int thdY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thdX < x && thdY < y)
    {
        // out[y][x]= in[y][x];
        out[thdY * x + thdX] = in[thdY * x + thdX];
    }
}

__global__ void copyRow(float *out, float *in, int x, int y)
{
    int thdX = blockIdx.x * blockDim.x + threadIdx.x;
    int thdY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thdX < x && thdY < y)
    {
        // out[y][x]= in[y][x];
        out[thdY * x + thdX] = in[thdY * x + thdX];
    }
}

__global__ void copyCol(float *out, float *in, int x, int y)
{
    int thdX = blockIdx.x * blockDim.x + threadIdx.x;
    int thdY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thdX < x && thdY < y)
    {
        // out[x][y] = in[x][y];
        out[thdX * y + thdY] = in[thdX * y + thdY];
    }
}

void checkResult(float *hostRef, float *gpuRef, const int size, int showme)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < size; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
                   gpuRef[i]);
            break;
        }

        if (showme && i > size / 2 && i < size / 2 + 5)
        {
            // printf("%dth element: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
        }
    }

    if (!match)
        printf("Arrays do not match.\n\n");
}

__global__ void transposeNaiveRow(float *out, float *in, int xx, int yy)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < xx && y < yy)
    {
        // out[x][y] = in[y][x]
        out[x * yy + y] = in[y * xx + x];
    }
}

__global__ void transposeNaiveCol(float *out, float *in, int xx, int yy)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < xx && y < yy)
    {
        out[y * xx + x] = in[x * yy + y];
    }
}

__global__ void transposeUnroll4Row(float *out, float *in, const int nx, const int ny) {
    
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting transpose at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    cudaSetDevice(dev);
    // set up array size 2048
    int nx = 1 << 11;
    int ny = 1 << 11;
    // select a kernel and block size
    int iKernel = 0;
    int blockx = 16;
    int blocky = 16;
    if (argc > 1)
        iKernel = atoi(argv[1]);
    if (argc > 2)
        blockx = atoi(argv[2]);
    if (argc > 3)
        blocky = atoi(argv[3]);
    if (argc > 4)
        nx = atoi(argv[4]);
    if (argc > 5)
        ny = atoi(argv[5]);
    printf(" with matrix nx %d ny %d with kernel %d\n", nx, ny, iKernel);
    size_t nBytes = nx * ny * sizeof(float);
    // execution configuration
    dim3 block(blockx, blocky);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    // allocate host memory
    float *h_A = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef = (float *)malloc(nBytes);
    // initialize host array
    initialData(h_A, nx * ny);
    // transpose at host side
    transposeHost(hostRef, h_A, nx, ny);
    // allocate device memory
    float *d_A, *d_C;
    cudaMalloc((void **)&d_A, nBytes);
    cudaMalloc((void **)&d_C, nBytes);
    // copy data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    // warmup to avoide startup overhead
    double iStart = seconds();
    warmup<<<grid, block>>>(d_C, d_A, nx, ny);
    cudaDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("warmup elapsed %f sec\n", iElaps);
    // kernel pointer and descriptor
    void (*kernel)(float *, float *, int, int);
    char *kernelName;
    // set up kernel
    switch (iKernel)
    {
    case 0:
        kernel = &copyRow;
        kernelName = "CopyRow ";
        break;
    case 1:
        kernel = &copyCol;
        kernelName = "CopyCol ";
        break;
    case 2:
        kernel = &transposeNaiveRow;
        kernelName = "NaiveRow ";
        break;
    case 3:
        kernel = &transposeNaiveCol;
        kernelName = "NaiveCol ";
        break;
    }
    // run kernel
    iStart = seconds();
    kernel<<<grid, block>>>(d_C, d_A, nx, ny);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    // calculate effective_bandwidth
    float ibnd = 2 * nx * ny * sizeof(float) / 1e9 / iElaps;
    printf("%s elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> "
           "effective bandwidth %f GB\n",
           kernelName, iElaps, grid.x, grid.y, block.x, block.y, ibnd);
    // check kernel results
    if (iKernel > 1)
    {
        cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
        checkResult(hostRef, gpuRef, nx * ny, 1);
    }
    // free host and device memory
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(hostRef);
    free(gpuRef);
    // reset device
    cudaDeviceReset();
    return EXIT_SUCCESS;
}