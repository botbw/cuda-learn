#include <stdio.h>

__global__ void hello_world() {
    printf("Hello from GPU\n");
}

int main() {
    printf("Hello from CPU\n");
    hello_world<<<1,10>>>();
    cudaDeviceSynchronize();
    return 0;
}
