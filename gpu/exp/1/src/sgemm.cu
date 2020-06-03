// taken from:
// https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf

// nvcc 036 sgemm .c -lcublas
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "nvmlPower.hpp"

#define IDX(i,j,ld) (((j)*(ld))+(i))
#ifndef RUNS
#define RUNS 1
#endif

int check(long, long, long, float*, float*, float*);

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
  B = (float*)malloc(K*N*sizeof(float)); 
  C = (float*)malloc(M*N*sizeof(float)); 

  for(j=0; j<K; j++) 
    for(i=0; i<M; i++) 
      A[IDX(i,j,M)] = (float) ((i*j+1) % K) / K;
  for(j=0; j<N; j++)
    for(i=0; i<K; i++) 
      B[IDX(i,j,K)] = (float) ((i*j+1) % N) / N;
  for(j=0; j<N; j++) 
    for(i=0; i<M; i++) 
      C[IDX(i,j,M)] = (float) ((i*j+1) % M) / M;

  // on the device
  float * d_A;
  float * d_B;
  float * d_C;
  cudaMalloc((void**)&d_A, M*K*sizeof(*A));
  cudaMalloc((void**)&d_B, K*N*sizeof(*B));
  cudaMalloc((void**)&d_C, M*N*sizeof(*C));
  cublasCreate(&handle); // initialize CUBLAS context

  // copy matrices from the host to the device
  cublasSetMatrix(M, K, sizeof(*A), A, M, d_A, M); //A -> d_A
  cublasSetMatrix(K, N, sizeof(*B), B, K, d_B, K); //B -> d_B
  cublasSetMatrix(M, N, sizeof(*C), C, M, d_C, M); //C -> d_C
  float alpha = 1.0;
  float beta = 1.0;
  


#ifndef CHECK
  // Invoke kernel for warm up
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
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
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
  checkCuda( cudaEventRecord(stopEvent, 0) );
  cudaDeviceSynchronize();
  nvmlAPIEnd();

  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );

  double time = ms/1000;

  double nFlops = (double)M*K*N*2*RUNS;
  double nFlopsPerSec = nFlops/time;
  double nGFlopsPerSec = nFlopsPerSec*1e-9;
  printf( "Time: %lf (sec), nFlops: %0.0lf, GFlopsS: %lf\n", time, nFlops, nGFlopsPerSec);
//  printf( "%lf, %0.2lf\n", time, nGFlopsPerSec);

  cublasGetMatrix(M, N, sizeof(*C), d_C, M, C, M); //d_C -> C
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cublasDestroy(handle);

  #ifdef CHECK
  int ret = check(M,N,K,A,B,C);
  #endif

  free (A);
  free (B);
  free (C);
  return EXIT_SUCCESS ;
}
