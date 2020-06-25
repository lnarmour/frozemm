// taken from:
// https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf

// nvcc 036 sgemm .c -lcublas
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "nvmlPower.hpp"
#include "cuda_profiler_api.h"
#include <math.h>

#define ceild(n,d)  (int)ceil(((double)(n))/((double)(d)))
#define floord(n,d) (int)floor(((double)(n))/((double)(d)))

#define IDX(i,j,ld) (((j)*(ld))+(i))
#ifndef RUNS
#define RUNS 1
#endif

int check(long, long, long, float*, float*, float*);

using namespace std;

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
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
  long M,N,K,PI,PJ,TK;
  if (argc>4) {
    M = N = K = atol(argv[1]);
    PI = atol(argv[2]);
    PJ = atol(argv[3]);
    TK = atol(argv[4]);
  } else {
    printf("Usage: ./MM N PI PJ TK\n");
    return 1;
  }

  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t *handle;
  cudaStream_t *stream;

  int num_streams = ceild(M,PI) * ceild(N,PJ);
  handle = (cublasHandle_t*)malloc(num_streams * sizeof(cublasHandle_t));
  stream = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
  for (int s=0; s<num_streams; s++) {
    cudaStreamCreate(&stream[s]);
    cublasCreate(&handle[s]); // initialize CUBLAS context
    cublasSetStream(handle[s], stream[s]);
  }

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
  checkCuda( cudaMalloc((void**)&d_A, M*K*sizeof(*A)) );
  checkCuda( cudaMalloc((void**)&d_B, K*N*sizeof(*B)) );
  checkCuda( cudaMalloc((void**)&d_C, M*N*sizeof(*C)) );

  // copy matrices from the host to the device
  stat = cublasSetMatrix(M, K, sizeof(*A), A, M, d_A, M); //A -> d_A
  stat = cublasSetMatrix(K, N, sizeof(*B), B, K, d_B, K); //B -> d_B
  stat = cublasSetMatrix(M, N, sizeof(*C), C, M, d_C, M); //C -> d_C
  float alpha = 1.0;
  float beta = 1.0;
  
  long pi,pj,tk,m,n,k;
  float *d_a, *d_b, *d_c;


#ifndef CHECK
  // Invoke kernel for warm up
  cublasSgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
#endif

  // Synchronize to make sure everyone is done in the warmup.
  cudaDeviceSynchronize();

  // events for timing
  cudaEvent_t startEvent, stopEvent;
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  float ms;
  int s;

  // ------------
  // time kernel
  // ------------

  nvmlAPIRun();
  checkCuda( cudaEventRecord(startEvent, 0) );

  cudaProfilerStart();
  for (tk=0; tk<K; tk+=TK) 
    for (pi=0; pi<M; pi+=PI)
      for (pj=0; pj<N; pj+=PJ) {
        s = ((int)(pi/PI)) * ceild(N,PJ) + (int)(pj/PJ);
        d_c = &(d_C[IDX(pi,pj,M)]);
        d_a = &(d_A[IDX(pi,tk,M)]);
        d_b = &(d_B[IDX(tk,pj,K)]);
        m = pi+PI<N ? PI : N-pi;
        n = pj+PJ<N ? PJ : N-pj;
        k = tk+TK<N ? TK : N-tk;
        stat = cublasSgemm(handle[s], CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_a, M, d_b, K, &beta, d_c, M);
      }
  cudaProfilerStop();

  checkCuda( cudaEventRecord(stopEvent, 0) );
  cudaDeviceSynchronize();
  nvmlAPIEnd();
  float energy;
  long total_ms;
  energy = nvmlAPI_getEnergy();
  total_ms = nvmlAPI_getTotalTime();
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );

  double flops = (double)M*K*N*2*RUNS;
  printf("ops:        %.2f GFLOPs\n", flops * 1e-9);
  printf("time:       %.5f sec\n", ms * 1e-3);
  printf("energy:     %.2f Joules\n", energy);
  printf("compute:    %.2f TFLOPs/sec\n", flops * 1e-12 / ((ms * 1e-3)));
  printf("avg power:  %.2f W\n", energy / (total_ms * 1e-3));

  stat = cublasGetMatrix(M, N, sizeof(*C), d_C, M, C, M); //d_C -> C
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  for (int s=0; s<num_streams; s++)
    cublasDestroy(handle[s]);

  #ifdef CHECK
  int ret = check(M,N,K,A,B,C);
  #endif

  free (A);
  free (B);
  free (C);
  free (stream);
  free (handle);
  return EXIT_SUCCESS ;
}
