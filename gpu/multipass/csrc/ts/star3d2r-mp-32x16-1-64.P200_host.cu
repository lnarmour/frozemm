#include <assert.h>
#include <stdio.h>
#include "star3d2r-mp-32x16-1-64.P200_kernel.hu"
#define BENCH_DIM 3
#define BENCH_FPP 25
#define BENCH_RAD 2
#define P 200

#include "common.h"

double kernel_stencil(SB_TYPE *A1, int compsize, int timestep, bool scop)
{
  double start_time = sb_time(), end_time = 0.0;
  int dimsize = compsize + BENCH_RAD * 2;
  SB_TYPE (*A)[dimsize][dimsize][dimsize]
    = (SB_TYPE (*)[dimsize][dimsize][dimsize])A1;

  int p = 1;

  if (scop) {
    if (timestep >= 1 && p >= 0 && p <= 10737417 && dimsize >= 200 * p + 1) {
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
#ifndef AN5D_TYPE
#define AN5D_TYPE unsigned
#endif
      const AN5D_TYPE __c0Len = (timestep - 0);
      const AN5D_TYPE __c0Pad = (0);
      #define __c0 c0
      const AN5D_TYPE __c1Len = (min(dimsize - 1, 200 * p + 199) - 200 * p + 1);
      const AN5D_TYPE __c1Pad = (200 * p);
      #define __c1 c1
      const AN5D_TYPE __c2Len = (min(dimsize - 1, 200 * p + 199) - 200 * p + 1);
      const AN5D_TYPE __c2Pad = (200 * p);
      #define __c2 c2
      const AN5D_TYPE __c3Len = (min(dimsize - 1, 200 * p + 199) - 200 * p + 1);
      const AN5D_TYPE __c3Pad = (200 * p);
      #define __c3 c3
      const AN5D_TYPE __halo1 = 2;
      const AN5D_TYPE __halo2 = 2;
      const AN5D_TYPE __halo3 = 2;
      AN5D_TYPE c0;
      AN5D_TYPE __side0LenMax;
      {
        const AN5D_TYPE __side0Len = 1;
        const AN5D_TYPE __side1Len = 64;
        const AN5D_TYPE __side2Len = 12;
        const AN5D_TYPE __side3Len = 28;
        const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
        const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
        const AN5D_TYPE __OlLen3 = (__halo3 * __side0Len);
        const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
        const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
        const AN5D_TYPE __side3LenOl = (__side3Len + 2 * __OlLen3);
        const AN5D_TYPE __blockSize = 1 * __side2LenOl * __side3LenOl;
        assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
        dim3 k0_dimBlock(__blockSize, 1, 1);
        dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len) * ((__c3Len + __side3Len - 1) / __side3Len), 1, 1);
        AN5D_TYPE __c0Padr = (__c0Len % 2) != (((__c0Len + __side0Len - 1) / __side0Len) % 2) && __c0Len % __side0Len < 2 ? 1 : 0;
        __side0LenMax = __side0Len;
        for (c0 = __c0Pad; c0 < __c0Pad + __c0Len / __side0Len - __c0Padr; c0 += 1)
        {
          kernel0_1<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, timestep, p, c0);
        }
      }
      if ((__c0Len % 2) != (((__c0Len + __side0LenMax - 1) / __side0LenMax) % 2))
      {
        if (__c0Len % __side0LenMax == 0)
        {
        }
      }
      else if (__c0Len % __side0LenMax)
      {
      }
    }
    cudaCheckKernel();
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
      for (int i = p*P; i < (p+1)*P; i++)
        for (int j = p*P; j < (p+1)*P; j++)
          for (int k = p*P; k < (p+1)*P; k++)
            A[(t+1)%2][i][j][k] =
              0.2500f * A[t%2][i][j][k]
              + 0.0620f * A[t%2][i-1][j][k] + 0.0621f * A[t%2][i+1][j][k]
              + 0.0622f * A[t%2][i][j-1][k] + 0.0623f * A[t%2][i][j+1][k]
              + 0.0624f * A[t%2][i][j][k-1] + 0.06245f * A[t%2][i][j][k+1]

              + 0.06255f * A[t%2][i-2][j][k] + 0.0626f * A[t%2][i+2][j][k]
              + 0.0627f  * A[t%2][i][j-2][k] + 0.0628f * A[t%2][i][j+2][k]
              + 0.0629f  * A[t%2][i][j][k-2] + 0.0630f * A[t%2][i][j][k+2];
  }

  return (((end_time != 0.0) ? end_time : sb_time()) - start_time);
}
