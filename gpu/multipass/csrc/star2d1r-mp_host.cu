#include <assert.h>
#include <stdio.h>
#include "star2d1r-mp_kernel.hu"
#define BENCH_DIM 2
#define BENCH_FPP 9
#define BENCH_RAD 1

#define PI 896
#define PJ 896

#include "common.h"

double kernel_stencil(SB_TYPE *A1, int compsize, int timestep, bool scop)
{
  double start_time = sb_time(), end_time = 0.0;
  int dimsize = compsize + BENCH_RAD * 2;
  SB_TYPE (*A)[dimsize][dimsize] = (SB_TYPE (*)[dimsize][dimsize])A1;

  if (scop) {
        if (dimsize >= 3 && timestep >= 1 && dimsize + timestep >= pi + 3 && pi >= -894 && pi <= 2147482751 && dimsize + timestep >= pj + 3 && dimsize + pj + 892 >= pi && dimsize + pi + 892 >= pj && pj >= -894 && pj <= 2147482751) {
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
    for (int pi=BENCH_RAD; pi<dimsize + (timestep-1)*BENCH_RAD; pi+=PI)
      for (int pj=BENCH_RAD; pj<dimsize + (timestep-1)*BENCH_RAD; pj+=PJ) {
    {
#ifndef AN5D_TYPE
#define AN5D_TYPE unsigned
#endif
      AN5D_TYPE c0 = 0;
      const AN5D_TYPE __c0Len = (min(min(timestep - 1, pi + 894), pj + 894) - max(max(0, -dimsize + pi + 2), -dimsize + pj + 2) + 1);
      const AN5D_TYPE __c0Pad = (max(max(0, -dimsize + pi + 2), -dimsize + pj + 2));
      #define __c0 c0
      const AN5D_TYPE __c1Len = (min(dimsize - 2, pi - c0 + 895) - max(1, pi - c0) + 1);
      const AN5D_TYPE __c1Pad = (max(1, pi - c0));
      #define __c1 c1
      const AN5D_TYPE __c2Len = (min(dimsize - 2, pj - c0 + 895) - max(1, pj - c0) + 1);
      const AN5D_TYPE __c2Pad = (max(1, pj - c0));
      #define __c2 c2
      const AN5D_TYPE __halo1 = 1;
      const AN5D_TYPE __halo2 = 1;
      AN5D_TYPE __side0LenMax;
      {
        const AN5D_TYPE __side0Len = 1;
        const AN5D_TYPE __side1Len = 128;
        const AN5D_TYPE __side2Len = 126;
        const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
        const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
        const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
        const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
        const AN5D_TYPE __blockSize = 1 * __side2LenOl;
        assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
        dim3 k0_dimBlock(__blockSize, 1, 1);
        dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
        AN5D_TYPE __c0Padr = (__c0Len % 2) != (((__c0Len + __side0Len - 1) / __side0Len) % 2) && __c0Len % __side0Len < 2 ? 1 : 0;
        __side0LenMax = __side0Len;
        for (c0 = __c0Pad; c0 < __c0Pad + __c0Len / __side0Len - __c0Padr; c0 += 1)
        {
          kernel0_1<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, timestep, pi, pj, c0);
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
        }
      }
    cudaCheckKernel();
{
#ifdef STENCILBENCH
cudaDeviceSynchronize();
SB_STOP_INSTRUMENTS;
#endif
          cudaCheckReturn(cudaMemcpy(A, dev_A, (size_t)(2) * (size_t)(dimsize) * (size_t)(dimsize) * sizeof(float), cudaMemcpyDeviceToHost));
}
          cudaCheckReturn(cudaFree(dev_A));
  }
  else {
    for (int pi=BENCH_RAD; pi<dimsize + (timestep-1)*BENCH_RAD; pi+=PI)
      for (int pj=BENCH_RAD; pj<dimsize + (timestep-1)*BENCH_RAD; pj+=PJ)
        for (int t = 0; t < timestep; t++) 
          for (int i = max(BENCH_RAD, pi + BENCH_RAD*(-t)); i < min(pi + PI + BENCH_RAD*(-t), dimsize-BENCH_RAD); i++)
            for (int j = max(BENCH_RAD, pj + BENCH_RAD*(-t)); j < min(pj + PJ + BENCH_RAD*(-t), dimsize-BENCH_RAD); j++)
              A[(t+1)%2][i][j] =
                A[t%2][i-1][j]
                + A[t%2][i][j-1]
                + A[t%2][i][j]
                + A[t%2][i][j+1]
                + A[t%2][i+1][j];
  }
  return (((end_time != 0.0) ? end_time : sb_time()) - start_time);
}
