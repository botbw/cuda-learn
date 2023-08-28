#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include "common.h"

typedef unsigned int uint;

// double seconds()
// {
//     struct timeval tp;
//     gettimeofday(&tp, NULL);
//     return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
// }

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

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, uint sz) {
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

__global__ void reduceInterleaved(int * g_idata, int *g_odata, unsigned int n) {
    uint tid = threadIdx.x;
    uint idx = blockIdx.x * blockDim.x + tid;
    if(idx >= n) return;
    for(int stride = blockDim.x / 2; stride > 0; stride = stride >> 1) {
        if(tid < stride) {
            g_idata[idx] += g_idata[idx + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
        g_odata[blockIdx.x] = g_idata[blockIdx.x * blockDim.x];
}

__global__ void reduceUnroll2(int *g_idata, int *g_odata, uint n) {
    uint tid = threadIdx.x;
    uint idx = 2 * blockIdx.x * blockDim.x + tid;
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;
    if (idx + blockDim.x < n) {
        idata[tid] += idata[tid + blockDim.x];
    }
    __syncthreads();
    for(int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        if(tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceUnrollWarp8(int * g_idata,int * g_odata,unsigned int n)
{
	//set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x*blockIdx.x*8+threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
	int *idata = g_idata + blockIdx.x*blockDim.x*8;
	//unrolling 8;
	if(idx+7 * blockDim.x<n)
	{
		int a1=g_idata[idx];
		int a2=g_idata[idx+blockDim.x];
		int a3=g_idata[idx+2*blockDim.x];
		int a4=g_idata[idx+3*blockDim.x];
		int a5=g_idata[idx+4*blockDim.x];
		int a6=g_idata[idx+5*blockDim.x];
		int a7=g_idata[idx+6*blockDim.x];
		int a8=g_idata[idx+7*blockDim.x];
		g_idata[idx]=a1+a2+a3+a4+a5+a6+a7+a8;

	}
	__syncthreads();
	//in-place reduction in global memory
	for (int stride = blockDim.x/2; stride>32; stride >>=1)
	{
		if (tid <stride)
		{
			idata[tid] += idata[tid + stride];
		}
		//synchronize within block
		__syncthreads();
	}
	//write result for this block to global mem
	if(tid<32)
	{
		volatile int *vmem = idata;
		// int *vmem = idata;
		vmem[tid]+=vmem[tid+32];
		vmem[tid]+=vmem[tid+16];
		vmem[tid]+=vmem[tid+8];
		vmem[tid]+=vmem[tid+4];
		vmem[tid]+=vmem[tid+2];
		vmem[tid]+=vmem[tid+1];

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
    // initialization
    int size = 1 << 24; // total number of elements to reduce
    printf(" with array size %d ", size);
    // execution configuration
    int blocksize = 128; // initial block size
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
    double iStart, iElaps;
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
    printf("cpu reduce elapsed %lf ms cpu_sum: %d\n", iElaps, cpu_sum);
    // kernel 0: warmup
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    iStart = seconds();
    warmup<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaPeekAtLastError());
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("gpu Warmup elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);
    // kernel 1: reduceNeighbored
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("gpu Neighbored elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);
    // kernel 2: reduceNeighboredLess
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    iStart = seconds();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("gpu reduceNeighboredLess elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);
    // kernel 3: reduceInterleaved
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    iStart = seconds();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("gpu reduceInterleaved elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);
    // kernel 4: reduceUnroll2
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    iStart = seconds();
    reduceUnroll2<<<grid.x / 2, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 2; i++)
        gpu_sum += h_odata[i];
    printf("gpu reduceUnroll2 elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x / 2, block.x);
    // kernel 5: reduceUnrollWarp8
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    iStart = seconds();
    reduceUnrollWarp8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += h_odata[i];
    printf("gpu reduceUnrollWarp8 elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x / 8, block.x);
    
    /// free host memory
    free(h_idata);
    free(h_odata);
    // free device memory
    cudaFree(d_idata);
    cudaFree(d_odata);
    // reset device
    cudaDeviceReset();
    return EXIT_SUCCESS;
}