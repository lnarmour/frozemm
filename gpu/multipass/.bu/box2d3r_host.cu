#include <assert.h>
#include <stdio.h>
#include "box2d3r_kernel.hu"
#define BENCH_DIM 2
#define BENCH_FPP 97
#define BENCH_RAD 3

#include "common.h"

#define PI 896
#define PJ 896

double kernel_stencil(SB_TYPE *A1, int compsize, int timestep, bool scop)
{
  double start_time = sb_time(), end_time = 0.0;
  int dimsize = compsize + BENCH_RAD * 2;
  SB_TYPE (*A)[dimsize][dimsize] = (SB_TYPE (*)[dimsize][dimsize])A1;

  int pi = 2;
  int pj = 2;

  if (scop) {
    if ((timestep >= 1 && pi >= 0 && pi <= 2396744 && dimsize >= 896 * pi + 1 && pj >= 0 && pj <= 2396744 && dimsize >= 896 * pj + 1) || (pj >= pi + 1 && pj <= 2396744 && dimsize + 3 * timestep >= 896 * pj + 4 && 896 * pj >= dimsize && ((-dimsize + 896 * pj) % 3) + dimsize + 896 * pi + 892 >= 896 * pj) || (dimsize >= 1 && pi <= 2396744 && dimsize + 3 * timestep >= 896 * pi + 4 && 896 * pi >= dimsize && pi >= pj && ((-dimsize + 896 * pi) % 3) + dimsize + 896 * pj + 892 >= 896 * pi)) {
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
#ifndef AN5D_TYPE
#define AN5D_TYPE unsigned
#endif
      const AN5D_TYPE __c0Len = (timestep - max(max(0, 299 * pi - (dimsize + pi + 2) / 3 + 1), 299 * pj - (dimsize + pj + 2) / 3 + 1));
      const AN5D_TYPE __c0Pad = (max(max(0, 299 * pi - (dimsize + pi + 2) / 3 + 1), 299 * pj - (dimsize + pj + 2) / 3 + 1));
      #define __c0 c0
      const AN5D_TYPE __c1Len = (min(dimsize - 1, 896 * pi - 3 * c0 + 895) - 896 * pi - 3 * c0 + 1);
      const AN5D_TYPE __c1Pad = (896 * pi - 3 * c0);
      #define __c1 c1
      const AN5D_TYPE __c2Len = (min(dimsize - 1, 896 * pj - 3 * c0 + 895) - 896 * pj - 3 * c0 + 1);
      const AN5D_TYPE __c2Pad = (896 * pj - 3 * c0);
      #define __c2 c2
      const AN5D_TYPE __halo1 = 3;
      const AN5D_TYPE __halo2 = 3;
      AN5D_TYPE c0;
      AN5D_TYPE __side0LenMax;
      {
        const AN5D_TYPE __side0Len = 2;
        const AN5D_TYPE __side1Len = 128;
        const AN5D_TYPE __side2Len = 244;
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
          kernel0_2<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, timestep, pi, pj, c0);
        }
      }
      if ((__c0Len % 2) != (((__c0Len + __side0LenMax - 1) / __side0LenMax) % 2))
      {
        if (__c0Len % __side0LenMax == 0)
        {
          {
            const AN5D_TYPE __side0Len = 1;
            const AN5D_TYPE __side1Len = 128;
            const AN5D_TYPE __side2Len = 250;
            const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
            const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
            const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
            const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
            const AN5D_TYPE __blockSize = 1 * __side2LenOl;
            assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
            dim3 k0_dimBlock(__blockSize, 1, 1);
            dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
            kernel0_1<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, timestep, pi, pj, c0);
          }
          c0 += 1;
          {
            const AN5D_TYPE __side0Len = 1;
            const AN5D_TYPE __side1Len = 128;
            const AN5D_TYPE __side2Len = 250;
            const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
            const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
            const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
            const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
            const AN5D_TYPE __blockSize = 1 * __side2LenOl;
            assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
            dim3 k0_dimBlock(__blockSize, 1, 1);
            dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
            kernel0_1<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, timestep, pi, pj, c0);
          }
        }
        else if (__c0Len % __side0LenMax == 1)
        {
          {
            const AN5D_TYPE __side0Len = 1;
            const AN5D_TYPE __side1Len = 128;
            const AN5D_TYPE __side2Len = 250;
            const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
            const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
            const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
            const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
            const AN5D_TYPE __blockSize = 1 * __side2LenOl;
            assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
            dim3 k0_dimBlock(__blockSize, 1, 1);
            dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
            kernel0_1<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, timestep, pi, pj, c0);
          }
          c0 += 1;
          {
            const AN5D_TYPE __side0Len = 1;
            const AN5D_TYPE __side1Len = 128;
            const AN5D_TYPE __side2Len = 250;
            const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
            const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
            const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
            const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
            const AN5D_TYPE __blockSize = 1 * __side2LenOl;
            assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
            dim3 k0_dimBlock(__blockSize, 1, 1);
            dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
            kernel0_1<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, timestep, pi, pj, c0);
          }
          c0 += 1;
          {
            const AN5D_TYPE __side0Len = 1;
            const AN5D_TYPE __side1Len = 128;
            const AN5D_TYPE __side2Len = 250;
            const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
            const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
            const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
            const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
            const AN5D_TYPE __blockSize = 1 * __side2LenOl;
            assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
            dim3 k0_dimBlock(__blockSize, 1, 1);
            dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
            kernel0_1<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, timestep, pi, pj, c0);
          }
        }
      }
      else if (__c0Len % __side0LenMax)
      {
        if (__c0Len % __side0LenMax == 1)
        {
          const AN5D_TYPE __side0Len = 1;
          const AN5D_TYPE __side1Len = 128;
          const AN5D_TYPE __side2Len = 250;
          const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
          const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
          const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
          const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
          const AN5D_TYPE __blockSize = 1 * __side2LenOl;
          assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
          dim3 k0_dimBlock(__blockSize, 1, 1);
          dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
          kernel0_1<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, timestep, pi, pj, c0);
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
  }
  else {
    for (int t = 0; t < timestep; t++)
#pragma omp parallel for
      for (int i = pi*PI - BENCH_RAD*t; i < (pi+1)*PI - BENCH_RAD*t; i++)
        for (int j = pj*PJ - BENCH_RAD*t; j < (pj+1)*PJ - BENCH_RAD*t; j++)
          A[(t+1)%2][i][j] =
            0.01530f * A[t%2][i-3][j-3] +
            0.01531f * A[t%2][i-3][j-2] +
            0.01532f * A[t%2][i-3][j-1] +
            0.01533f * A[t%2][i-3][j] +
            0.01534f * A[t%2][i-3][j+1] +
            0.01535f * A[t%2][i-3][j+2] +
            0.01536f * A[t%2][i-3][j+3] +

            0.01537f * A[t%2][i-2][j-3] +
            0.01538f * A[t%2][i-2][j-2] +
            0.01539f * A[t%2][i-2][j-1] +
            0.01540f * A[t%2][i-2][j] +
            0.01541f * A[t%2][i-2][j+1] +
            0.01542f * A[t%2][i-2][j+2] +
            0.01543f * A[t%2][i-2][j+3] +

            0.01544f * A[t%2][i-1][j-3] +
            0.01545f * A[t%2][i-1][j-2] +
            0.01546f * A[t%2][i-1][j-1] +
            0.01546f * A[t%2][i-1][j] +
            0.01547f * A[t%2][i-1][j+1] +
            0.01548f * A[t%2][i-1][j+2] +
            0.01549f * A[t%2][i-1][j+3] +

            0.01550f * A[t%2][i][j-3] +
            0.01551f * A[t%2][i][j-2] +
            0.01552f * A[t%2][i][j-1] +
            0.25424f * A[t%2][i][j] +
            0.01554f * A[t%2][i][j+1] +
            0.01555f * A[t%2][i][j+2] +
            0.01556f * A[t%2][i][j+3] +

            0.01557f * A[t%2][i+1][j-3] +
            0.01558f * A[t%2][i+1][j-2] +
            0.01559f * A[t%2][i+1][j-1] +
            0.01560f * A[t%2][i+1][j] +
            0.01561f * A[t%2][i+1][j+1] +
            0.01562f * A[t%2][i+1][j+2] +
            0.01564f * A[t%2][i+1][j+3] +

            0.01565f * A[t%2][i+2][j-3] +
            0.01566f * A[t%2][i+2][j-2] +
            0.01567f * A[t%2][i+2][j-1] +
            0.01568f * A[t%2][i+2][j] +
            0.01569f * A[t%2][i+2][j+1] +
            0.01570f * A[t%2][i+2][j+2] +
            0.01571f * A[t%2][i+2][j+3] +

            0.01572f * A[t%2][i+3][j-3] +
            0.01573f * A[t%2][i+3][j-2] +
            0.01574f * A[t%2][i+3][j-1] +
            0.01575f * A[t%2][i+3][j] +
            0.01576f * A[t%2][i+3][j+1] +
            0.01577f * A[t%2][i+3][j+2] +
            0.01578f * A[t%2][i+3][j+3];
  }

  return (((end_time != 0.0) ? end_time : sb_time()) - start_time);
}
