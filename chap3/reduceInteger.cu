#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

typedef unsigned int uint;

double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

int recursiveReduce(int *data, int const size)
{
    // terminate check
    if (size == 1)
        return data[0];
    // renew the stride
    int const stride = size / 2;
    if (size % 2 == 1)
    {
        for (int i = 0; i < stride; i++)
        {
            data[i] += data[i + stride];
        }
        data[0] += data[size - 1];
    }
    else
    {
        for (int i = 0; i < stride; i++)
        {
            data[i] += data[i + stride];
        }
    }
    // call
    return recursiveReduce(data, stride);
}

__global__ void reduceNeighbored(int *g_idata, int *g_odata, uint sz)
{
    uint tid = threadIdx.x;
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sz)
        return;
    int *idata = g_idata + blockIdx.x * blockDim.x;
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (tid % (stride * 2) == 0)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeightboredLess(int *g_idata, int *g_odata, uint sz) {
    uint tid = threadIdx.x;
    uint idx = blockIdx.x * blockDim.x + tid;
    if(idx >= sz) return;
    int *idata = g_idata + blockIdx.x * blockDim.x;
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int id = tid * 2 * stride;
        if (id < blockDim.x)
		{
			idata[id] += idata[id + stride];
		}
        __syncthreads();
    }
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void warmup(int *g_idata, int *g_odata, uint sz)
{
    uint tid = threadIdx.x;
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sz)
        return;
    int *idata = g_idata + blockIdx.x * blockDim.x;
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (tid % (stride * 2) == 0)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    cudaSetDevice(dev);
    bool bResult = false;
    // initialization
    int size = 1 << 24; // total number of elements to reduce
    printf(" with array size %d ", size);
    // execution configuration
    int blocksize = 512; // initial block size
    if (argc > 1)
    {
        blocksize = atoi(argv[1]); // block size from command line argument
    }
    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);
    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(bytes);
    // initialize the array
    for (int i = 0; i < size; i++)
    {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = 1;
    }
    memcpy(tmp, h_idata, bytes);
    size_t iStart, iElaps;
    int gpu_sum = 0;
    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **)&d_idata, bytes);
    cudaMalloc((void **)&d_odata, grid.x * sizeof(int));
    // cpu reduction
    iStart = seconds();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce elapsed %d ms cpu_sum: %d\n", iElaps, cpu_sum);
    // kernel 1: reduceNeighbored
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    warmup<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("gpu Warmup elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);
    // kernel 1: reduceNeighbored
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("gpu Neighbored elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);
    /// free host memory
    free(h_idata);
    free(h_odata);
    // free device memory
    cudaFree(d_idata);
    cudaFree(d_odata);
    // reset device
    cudaDeviceReset();
    // check the results
    bResult = (gpu_sum == cpu_sum);
    if (!bResult)
        printf("Test failed!\n");
    return EXIT_SUCCESS;
}