// taken from:
// https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf

// nvcc 036 sgemm .c -lcublas
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "timer.h"

#define IDX(i,j,ld) (((j)*(ld))+(i))

int check(long, long, long, float*, float*, float*);

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
  cublasHandle_t handle;

  int i,j; 
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
  cudaStat = cudaMalloc((void**)&d_A, M*K*sizeof(*A));
  cudaStat = cudaMalloc((void**)&d_B, K*N*sizeof(*B));
  cudaStat = cudaMalloc((void**)&d_C, M*N*sizeof(*C));
  stat = cublasCreate(&handle); // initialize CUBLAS context

  // copy matrices from the host to the device
  stat = cublasSetMatrix(M, K, sizeof(*A), A, M, d_A, M); //A -> d_A
  stat = cublasSetMatrix(K, N, sizeof(*B), B, K, d_B, K); //B -> d_B
  stat = cublasSetMatrix(M, N, sizeof(*C), C, M, d_C, M); //C -> d_C
  float alpha = 1.0;
  float beta = 1.0;
  
  initialize_timer();
  start_timer();

  // cblasSgemm -> C[i,j] += alpha*A[i,k]*B[k,j] + beta*C[i,j]
  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);

  //cudaThreadSynchronize();

  stop_timer();
  double time = elapsed_time();

  double nFlops = (double)M*K*N*2;
  double nFlopsPerSec = nFlops/time;
  double nGFlopsPerSec = nFlopsPerSec*1e-9;
  printf( "Time: %lf (sec), nFlops: %0.0lf, GFlopsS: %lf\n", time, nFlops, nGFlopsPerSec);

  stat = cublasGetMatrix(M, N, sizeof(*C), d_C, M, C, M); //d_C -> C
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
