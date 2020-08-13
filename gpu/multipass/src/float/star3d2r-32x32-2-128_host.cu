#include <assert.h>
#include <stdio.h>
#include "star3d2r-32x32-2-128_kernel.hu"
#define BENCH_DIM 3
#define BENCH_FPP 25
#define BENCH_RAD 2

#define pi 3
#define pj 3
#define pk 3
#define PI 96
#define PJ 96
#define PK 96

#include "common.h"

double kernel_stencil(SB_TYPE *A1, int compsize, int timestep, bool scop)
{
  double start_time = sb_time(), end_time = 0.0;
  int dimsize = compsize + BENCH_RAD * 2;
  SB_TYPE (*A)[dimsize][dimsize][dimsize]
    = (SB_TYPE (*)[dimsize][dimsize][dimsize])A1;

  if (scop) {
    if (timestep >= 1 && dimsize + 2 * timestep >= 293) {
#define cudaCheckReturn(ret) \
  do { \
    cudaError_t cudaCheckReturn_e = (ret); \
    if (cudaCheckReturn_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaCheckReturn_e)); \
      fflush(stderr); \
    } \
    assert(cudaCheckReturn_e == cudaSuccess); \
  } while(0)
#define cudaCheckKernel() \
  do { \
    cudaCheckReturn(cudaGetLastError()); \
  } while(0)

      float *dev_A;
      
      cudaCheckReturn(cudaMalloc((void **) &dev_A, (size_t)(2) * (size_t)(dimsize) * (size_t)(dimsize) * (size_t)(dimsize) * sizeof(float)));
      
{
      cudaCheckReturn(cudaMemcpy(dev_A, A, (size_t)(2) * (size_t)(dimsize) * (size_t)(dimsize) * (size_t)(dimsize) * sizeof(float), cudaMemcpyHostToDevice));
#ifdef STENCILBENCH
cudaDeviceSynchronize();
SB_START_INSTRUMENTS;
#endif
}
      {
        dim3 k0_dimBlock;
        dim3 k0_dimGrid;
        {
          kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, timestep);
          cudaCheckKernel();
        }
      }
      
{
#ifdef STENCILBENCH
cudaDeviceSynchronize();
SB_STOP_INSTRUMENTS;
#endif
      cudaCheckReturn(cudaMemcpy(A, dev_A, (size_t)(2) * (size_t)(dimsize) * (size_t)(dimsize) * (size_t)(dimsize) * sizeof(float), cudaMemcpyDeviceToHost));
}
      cudaCheckReturn(cudaFree(dev_A));
    }
  }
  else {
    for (int t = 0; t < timestep; t++)
#pragma omp parallel for
      for (int i = BENCH_RAD; i < dimsize - BENCH_RAD; i++)
        for (int j = BENCH_RAD; j < dimsize - BENCH_RAD; j++)
          for (int k = BENCH_RAD; k < dimsize - BENCH_RAD; k++)
            A[(t+1)%2][i-BENCH_RAD*t][j-BENCH_RAD*t][k-BENCH_RAD*t] =
              0.2500f * A[t%2][i-BENCH_RAD*t][j-BENCH_RAD*t][k-BENCH_RAD*t]
              + 0.0620f * A[t%2][i-1][j][k] + 0.0621f * A[t%2][i+1-BENCH_RAD*t][j-BENCH_RAD*t][k-BENCH_RAD*t]
              + 0.0622f * A[t%2][i][j-1][k] + 0.0623f * A[t%2][i-BENCH_RAD*t][j+1-BENCH_RAD*t][k-BENCH_RAD*t]
              + 0.0624f * A[t%2][i][j][k-1] + 0.06245f * A[t%2][i-BENCH_RAD*t][j-BENCH_RAD*t][k+1-BENCH_RAD*t]

              + 0.06255f * A[t%2][i-2][j][k] + 0.0626f * A[t%2][i+2-BENCH_RAD*t][j-BENCH_RAD*t][k-BENCH_RAD*t]
              + 0.0627f  * A[t%2][i][j-2][k] + 0.0628f * A[t%2][i-BENCH_RAD*t][j+2-BENCH_RAD*t][k-BENCH_RAD*t]
              + 0.0629f  * A[t%2][i][j][k-2] + 0.0630f * A[t%2][i-BENCH_RAD*t][j-BENCH_RAD*t][k+2-BENCH_RAD*t];
  }

  return (((end_time != 0.0) ? end_time : sb_time()) - start_time);
}
