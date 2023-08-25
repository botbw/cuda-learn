#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    printf("------------------------------------------------------\n");

    int devCnt = 0;
    cudaError_t error = cudaGetDeviceCount(&devCnt);
    if(error != cudaSuccess) {
        printf("cudaGetDeviceCount failed! %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    if(devCnt) {
        printf("Found %d CUDA devices\n", devCnt);
    } else {
        printf("No CUDA devices found\n");
    }

    int dev = devCnt - 1;
    cudaSetDevice(dev);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device %d: %s\n", dev, prop.name);

    int cudaDriverVersion, cudaRuntimeVersion;
    cudaDriverGetVersion(&cudaDriverVersion);
    cudaRuntimeGetVersion(&cudaRuntimeVersion);

    printf("Raw CUDA Driver Version / Runtime Version: %d / %d\n",
        cudaDriverVersion, cudaRuntimeVersion);
    printf("CUDA Driver Version / Runtime Version: %d.%d / %d.%d\n",
        cudaDriverVersion / 1000, (cudaDriverVersion % 100) / 10,
        cudaRuntimeVersion / 1000, (cudaRuntimeVersion % 100) / 10);
    printf("CUDA Capability Major/Minor version number: %d.%d\n", prop.major, prop.minor);
    printf("Total amount of global memory: %.2f GBytes (%llu bytes)\n",
        (float)prop.totalGlobalMem / pow(1024.0, 3),
        (unsigned long long)prop.totalGlobalMem);
    printf("GPU Clock rate: %.0f MHz (%0.2f GHz)\n", prop.clockRate * 1e-3f, prop.clockRate * 1e-6f);
    printf("Memory Clock rate: %.0f Mhz\n", prop.memoryClockRate * 1e-3f);
    printf("Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
    if(prop.l2CacheSize) {
        printf("L2 Cache Size: %d bytes\n", prop.l2CacheSize);
    }
    printf("Max Texture Dimension Size (x,y,z): 1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
        prop.maxTexture1D, prop.maxTexture2D[0], prop.maxTexture2D[1],
        prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
    printf("Max Layered Texture Size (dim) x layers: 1D=(%d) x %d, 2D=(%d,%d) x %d\n",
        prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1],
        prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1],
        prop.maxTexture2DLayered[2]);
    printf("Total amount of constant memory: %lu bytes\n", prop.totalConstMem);
    printf("Total amount of shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
    printf("Total number of registers available per block: %d\n", prop.regsPerBlock);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Maximum number of threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Number of multiprocessors on device: %d\n", prop.multiProcessorCount);
    printf("Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Maximum sizes of each dimension of a block: %d x %d x %d\n",
        prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Maximum sizes of each dimension of a grid: %d x %d x %d\n",
        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Maximum memory pitch: %lu bytes\n", prop.memPitch);
    printf("------------------------------------------------------\n");
    printf("Texture alignment: %lu bytes\n", prop.textureAlignment);
    printf("Concurrent copy and execution: %s with %d copy engine(s)\n",
        (prop.deviceOverlap ? "Yes" : "No"), prop.asyncEngineCount);
    printf("Run time limit on kernels: %s\n", prop.kernelExecTimeoutEnabled ? "Yes" : "No");
    printf("Integrated GPU sharing Host Memory: %s\n", prop.integrated ? "Yes" : "No");
    printf("Support host page-locked memory mapping: %s\n", prop.canMapHostMemory ? "Yes" : "No");
    printf("Alignment requirement for Surfaces: %s\n", prop.surfaceAlignment ? "Yes" : "No");
    printf("Device has ECC support: %s\n", prop.ECCEnabled ? "Enabled" : "Disabled");
    printf("Device supports Unified Addressing (UVA): %s\n", prop.unifiedAddressing ? "Yes" : "No");
    printf("Device supports Compute Preemption: %s\n", prop.computePreemptionSupported ? "Yes" : "No");
    printf("Supports Cooperative Kernel Launch: %s\n", prop.cooperativeLaunch ? "Yes" : "No");
    printf("Supports MultiDevice Co-op Kernel Launch: %s\n", prop.cooperativeMultiDeviceLaunch ? "Yes" : "No");
    printf("Device PCI Domain ID / Bus ID / location ID: %d / %d / %d\n",
        prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    const char *sComputeMode[] = {
        "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
        "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
        "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
        "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
        "Unknown",
        NULL
    };
    printf("Compute Mode: %s\n", sComputeMode[prop.computeMode]);
    printf("Concurrent kernel execution: %s\n", prop.concurrentKernels ? "Yes" : "No");
    printf("Device has ECC support enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
    printf("Device is using TCC driver mode: %s\n", prop.tccDriver ? "Yes" : "No");
    printf("Device supports Unified Addressing (UVA): %s\n", prop.unifiedAddressing ? "Yes" : "No");
    printf("Device PCI Domain ID / Bus ID / location ID: %d / %d / %d\n",
        prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    printf("Device supports Compute Preemption: %s\n", prop.computePreemptionSupported ? "Yes" : "No");
    printf("Supports Cooperative Kernel Launch: %s\n", prop.cooperativeLaunch ? "Yes" : "No");
    printf("Supports MultiDevice Co-op Kernel Launch: %s\n", prop.cooperativeMultiDeviceLaunch ? "Yes" : "No");
    printf("Device PCI Domain ID / Bus ID / location ID: %d / %d / %d\n",
        prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    exit(EXIT_SUCCESS);
}