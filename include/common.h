#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <stdio.h>

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

#endif 