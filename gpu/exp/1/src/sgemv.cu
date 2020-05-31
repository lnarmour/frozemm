// taken from:
// https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf
// tweaked to do matrix-vector product

// nvcc 036 sgemm .c -lcublas
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "nvmlPower.hpp"

#define IDX(i,j,ld) (((j)*(ld))+(i))
#define RUNS 1

int checkmv(long, long, float*, float*, float*);

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
  long M,N,K;
  if (argc>1) {
    M = N = K = atol(argv[1]);
  } else {
    printf("Usage: %s N\n", argv[0]);
    return 1;
  }

  cublasHandle_t handle;

  long i,j; 
  float* A;
  float* B;
  float* C;
  A = (float*)malloc(M*K*sizeof(float)); 
  B = (float*)malloc(K*sizeof(float)); 
  C = (float*)malloc(M*sizeof(float)); 

  for(j=0; j<K; j++) 
    for(i=0; i<M; i++) 
      A[IDX(i,j,M)] = (float) ((i*j+1) % K) / K;
  for(i=0; i<K; i++) 
    B[i] = (float) ((i+1) % N) / N;
  for(i=0; i<M; i++) 
    C[i] = 0.0;

  // on the device
  float * d_A;
  float * d_B;
  float * d_C;
  cudaMalloc((void**)&d_A, M*K*sizeof(*A));
  cudaMalloc((void**)&d_B, K*sizeof(*B));
  cudaMalloc((void**)&d_C, M*sizeof(*C));
  cublasCreate(&handle); // initialize CUBLAS context

  // copy matrices from the host to the device
  cublasSetMatrix(M, K, sizeof(*A), A, M, d_A, M); //A -> d_A
  cublasSetVector(K, sizeof(*B), B, 1, d_B, 1); //B -> d_B
  cublasSetVector(M, sizeof(*C), C, 1, d_C, 1); //C -> d_C
  float alpha = 1.0;
  float beta = 1.0;
  


#ifndef CHECK
  // Invoke kernel for warm up
  cublasSgemv(handle, CUBLAS_OP_N, M, K, &alpha, d_A, M, d_B, 1, &beta, d_C, 1);
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
    cublasSgemv(handle, CUBLAS_OP_N, M, K, &alpha, d_A, M, d_B, 1, &beta, d_C, 1);
  checkCuda( cudaEventRecord(stopEvent, 0) );
  cudaDeviceSynchronize();
  nvmlAPIEnd();

  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );

  double time = ms/1000;

  double nFlops = (double)M*K*2*RUNS;
  double nFlopsPerSec = nFlops/time;
  double nGFlopsPerSec = nFlopsPerSec*1e-9;
  printf( "Time: %lf (sec), nFlops: %0.0lf, GFlopsS: %lf\n", time, nFlops, nGFlopsPerSec);

  cublasGetVector(M, sizeof(*C), d_C, 1, C, 1); //d_C -> C
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cublasDestroy(handle);

  #ifdef CHECK
  int ret = checkmv(M,K,A,B,C);
  #endif

  free (A);
  free (B);
  free (C);
  return EXIT_SUCCESS ;
}
