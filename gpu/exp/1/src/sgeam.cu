// taken from:
// https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf

// nvcc 036 sgemm .c -lcublas
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "nvmlPower.hpp"

#define IDX(i,j,ld) (((j)*(ld))+(i))
#define RUNS 1000

int checkt(long, long, float*, float*, float*);

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

int main (int argc, char** argv) {
  long M,N;
  if (argc>1) {
    M = N = atol(argv[1]);
  } else {
    printf("Usage: %s N\n", argv[0]);
    return 1;
  }

  cublasHandle_t handle;

  long i,j; 
  float* A;
  float* B;
  float* C;
  A = (float*)malloc(M*N*sizeof(float)); 
  B = (float*)malloc(M*N*sizeof(float)); 
  C = (float*)malloc(M*N*sizeof(float)); 

  for(j=0; j<N; j++) 
    for(i=0; i<M; i++) 
      A[IDX(i,j,M)] = (float) ((i*j+1) % M) / M;

  // on the device
  float * d_A;
  float * d_B;
  float * d_C;
  cudaMalloc((void**)&d_A, M*N*sizeof(*A));
  cudaMalloc((void**)&d_B, M*N*sizeof(*B));
  cudaMalloc((void**)&d_C, M*N*sizeof(*C));
  cublasCreate(&handle); // initialize CUBLAS context

  // copy matrices from the host to the device
  cublasSetMatrix(M, N, sizeof(*A), A, M, d_A, M); //A -> d_A
  cublasSetMatrix(M, N, sizeof(*B), B, M, d_B, M); //B -> d_B
  cublasSetMatrix(M, N, sizeof(*C), C, M, d_C, M); //C -> d_C
  float alpha = 1.0;
  float beta = 0.0;
  
#ifndef CHECK
  // Invoke kernel for warm up
  cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, &alpha, d_A, M, &beta, d_B, M, d_C, M);
#endif

  // Synchronize to make sure everyone is done in the warmup.
  cudaDeviceSynchronize();

  // events for timing
  cudaEvent_t startEvent, stopEvent;
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  float ms;

  // ------------
  // time kernel
  // ------------

  nvmlAPIRun();
  checkCuda( cudaEventRecord(startEvent, 0) );
  for (int r=0; r<RUNS; r++)
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, &alpha, d_A, M, &beta, d_B, M, d_C, M);
  checkCuda( cudaEventRecord(stopEvent, 0) );
  cudaDeviceSynchronize();
  nvmlAPIEnd();

  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );

  double time = ms/1000;

  printf( "Time: %lf (sec)\n", time);

  cublasGetMatrix(M, N, sizeof(*C), d_C, M, C, M); //d_C -> C
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cublasDestroy(handle);

  #ifdef CHECK
  int ret = checkt(M,N,A,B,C);
  #endif

  free (A);
  free (B);
  free (C);
  return EXIT_SUCCESS ;
}
