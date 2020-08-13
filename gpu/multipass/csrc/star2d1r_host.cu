#include <assert.h>
#include <stdio.h>
#include "star2d1r_kernel.hu"
#define BENCH_DIM 2
#define BENCH_FPP 9
#define BENCH_RAD 1

#include "common.h"

#define PI 512
#define PJ 512
#define pi 2
#define pj 2

double kernel_stencil(SB_TYPE *A1, int compsize, int timestep, bool scop)
{
  double start_time = sb_time(), end_time = 0.0;
  int dimsize = compsize + BENCH_RAD * 2;
  SB_TYPE (*A)[dimsize][dimsize] = (SB_TYPE (*)[dimsize][dimsize])A1;

  if (scop) {
    if (dimsize + timestep >= 1027 && timestep >= 1) {
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
      
      cudaCheckReturn(cudaMalloc((void **) &dev_A, (size_t)(2) * (size_t)(dimsize) * (size_t)(dimsize) * sizeof(float)));
      
{
      cudaCheckReturn(cudaMemcpy(dev_A, A, (size_t)(2) * (size_t)(dimsize) * (size_t)(dimsize) * sizeof(float), cudaMemcpyHostToDevice));
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
      cudaCheckReturn(cudaMemcpy(A, dev_A, (size_t)(2) * (size_t)(dimsize) * (size_t)(dimsize) * sizeof(float), cudaMemcpyDeviceToHost));
}
      cudaCheckReturn(cudaFree(dev_A));
    }
  }
  else {
    for (int t = 0; t < timestep; t++)
#pragma omp parallel for
      for (int i = pi*PI+BENCH_RAD; i < (pi+1)*PI-BENCH_RAD; i++)
        for (int j = pj*PJ+BENCH_RAD; j < (pj+1)*PJ-BENCH_RAD; j++)
          A[(t+1)%2][i-BENCH_RAD*t][j-BENCH_RAD*t] =
            0.1873f * A[t%2][i-1-BENCH_RAD*t][j-BENCH_RAD*t]
            + 0.1876f * A[t%2][i-BENCH_RAD*t][j-1-BENCH_RAD*t]
            + 0.2500f * A[t%2][i-BENCH_RAD*t][j-BENCH_RAD*t]
            + 0.1877f * A[t%2][i-BENCH_RAD*t][j+1-BENCH_RAD*t]
            + 0.1874f * A[t%2][i+1-BENCH_RAD*t][j-BENCH_RAD*t];
  }

  return (((end_time != 0.0) ? end_time : sb_time()) - start_time);
}
