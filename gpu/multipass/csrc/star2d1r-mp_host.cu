#include <assert.h>
#include <stdio.h>
#include "star2d1r-mp_kernel.hu"
#define BENCH_DIM 2
#define BENCH_FPP 5
#define BENCH_RAD 1

#define PI 896
#define PJ 896

#include "common.h"

void step_pass(int dimsize, SB_TYPE (*X)[dimsize][dimsize], int pi, int pj, int bt, int T);

double kernel_stencil(SB_TYPE *A1, int compsize, int timestep, bool scop)
{
  double start_time = sb_time(), end_time = 0.0;
  int dimsize = compsize + BENCH_RAD * 2;
  SB_TYPE (*A)[dimsize][dimsize] = (SB_TYPE (*)[dimsize][dimsize])A1;

  int pi = 2*PI;
  int pj = 2*PJ;

  if (scop) {
    int BT = 4;

      if (dimsize >= 1 && BT >= 1 && dimsize >= pi + 1 && pi >= -895 && pi <= 2147482751 && dimsize >= pj + 1 && pj >= -895 && pj <= 2147482751) {
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
        float *dev_A_root;
        
        cudaCheckReturn(cudaMalloc((void **) &dev_A_root, (size_t)(2) * (size_t)(dimsize) * (size_t)(dimsize) * sizeof(float)));
        
{
        cudaCheckReturn(cudaMemcpy(dev_A_root, A, (size_t)(2) * (size_t)(dimsize) * (size_t)(dimsize) * sizeof(float), cudaMemcpyHostToDevice));
#ifdef STENCILBENCH
cudaDeviceSynchronize();
SB_START_INSTRUMENTS;
#endif
}
    for (int bt=0; bt<timestep; bt+=BT) {

      dev_A = &(dev_A_root[dimsize*dimsize + (pi-BENCH_RAD*bt)*dimsize + pj-BENCH_RAD*bt]);

    {
#ifndef AN5D_TYPE
#define AN5D_TYPE unsigned
#endif
      const AN5D_TYPE __c0Len = (bt + BT - bt);
      const AN5D_TYPE __c0Pad = (bt);
      #define __c0 c0
      const AN5D_TYPE __c1Len = (min(dimsize - 1, pi + 895) - pi + 1);
      const AN5D_TYPE __c1Pad = (pi);
      #define __c1 c1
      const AN5D_TYPE __c2Len = (min(dimsize - 1, pj + 895) - pj + 1);
      const AN5D_TYPE __c2Pad = (pj);
      #define __c2 c2
      const AN5D_TYPE __halo1 = 1;
      const AN5D_TYPE __halo2 = 1;
      AN5D_TYPE c0;
      AN5D_TYPE __side0LenMax;
      {
        const AN5D_TYPE __side0Len = 4;
        const AN5D_TYPE __side1Len = 128;
        const AN5D_TYPE __side2Len = 24;
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
          kernel0_4<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, bt, BT, pi, pj, c0);
        }
      }
      if ((__c0Len % 2) != (((__c0Len + __side0LenMax - 1) / __side0LenMax) % 2))
      {
        if (__c0Len % __side0LenMax == 0)
        {
          {
            const AN5D_TYPE __side0Len = 2;
            const AN5D_TYPE __side1Len = 128;
            const AN5D_TYPE __side2Len = 28;
            const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
            const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
            const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
            const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
            const AN5D_TYPE __blockSize = 1 * __side2LenOl;
            assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
            dim3 k0_dimBlock(__blockSize, 1, 1);
            dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
            kernel0_2<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, bt, BT, pi, pj, c0);
          }
          c0 += 1;
          {
            const AN5D_TYPE __side0Len = 2;
            const AN5D_TYPE __side1Len = 128;
            const AN5D_TYPE __side2Len = 28;
            const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
            const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
            const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
            const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
            const AN5D_TYPE __blockSize = 1 * __side2LenOl;
            assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
            dim3 k0_dimBlock(__blockSize, 1, 1);
            dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
            kernel0_2<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, bt, BT, pi, pj, c0);
          }
        }
        else if (__c0Len % __side0LenMax == 1)
        {
          {
            const AN5D_TYPE __side0Len = 3;
            const AN5D_TYPE __side1Len = 128;
            const AN5D_TYPE __side2Len = 26;
            const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
            const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
            const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
            const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
            const AN5D_TYPE __blockSize = 1 * __side2LenOl;
            assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
            dim3 k0_dimBlock(__blockSize, 1, 1);
            dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
            kernel0_3<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, bt, BT, pi, pj, c0);
          }
          c0 += 1;
          {
            const AN5D_TYPE __side0Len = 1;
            const AN5D_TYPE __side1Len = 128;
            const AN5D_TYPE __side2Len = 30;
            const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
            const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
            const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
            const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
            const AN5D_TYPE __blockSize = 1 * __side2LenOl;
            assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
            dim3 k0_dimBlock(__blockSize, 1, 1);
            dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
            kernel0_1<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, bt, BT, pi, pj, c0);
          }
          c0 += 1;
          {
            const AN5D_TYPE __side0Len = 1;
            const AN5D_TYPE __side1Len = 128;
            const AN5D_TYPE __side2Len = 30;
            const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
            const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
            const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
            const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
            const AN5D_TYPE __blockSize = 1 * __side2LenOl;
            assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
            dim3 k0_dimBlock(__blockSize, 1, 1);
            dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
            kernel0_1<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, bt, BT, pi, pj, c0);
          }
        }
        else if (__c0Len % __side0LenMax == 2)
        {
          {
            const AN5D_TYPE __side0Len = 1;
            const AN5D_TYPE __side1Len = 128;
            const AN5D_TYPE __side2Len = 30;
            const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
            const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
            const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
            const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
            const AN5D_TYPE __blockSize = 1 * __side2LenOl;
            assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
            dim3 k0_dimBlock(__blockSize, 1, 1);
            dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
            kernel0_1<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, bt, BT, pi, pj, c0);
          }
          c0 += 1;
          {
            const AN5D_TYPE __side0Len = 1;
            const AN5D_TYPE __side1Len = 128;
            const AN5D_TYPE __side2Len = 30;
            const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
            const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
            const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
            const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
            const AN5D_TYPE __blockSize = 1 * __side2LenOl;
            assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
            dim3 k0_dimBlock(__blockSize, 1, 1);
            dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
            kernel0_1<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, bt, BT, pi, pj, c0);
          }
        }
        else if (__c0Len % __side0LenMax == 3)
        {
          {
            const AN5D_TYPE __side0Len = 2;
            const AN5D_TYPE __side1Len = 128;
            const AN5D_TYPE __side2Len = 28;
            const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
            const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
            const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
            const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
            const AN5D_TYPE __blockSize = 1 * __side2LenOl;
            assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
            dim3 k0_dimBlock(__blockSize, 1, 1);
            dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
            kernel0_2<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, bt, BT, pi, pj, c0);
          }
          c0 += 1;
          {
            const AN5D_TYPE __side0Len = 1;
            const AN5D_TYPE __side1Len = 128;
            const AN5D_TYPE __side2Len = 30;
            const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
            const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
            const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
            const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
            const AN5D_TYPE __blockSize = 1 * __side2LenOl;
            assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
            dim3 k0_dimBlock(__blockSize, 1, 1);
            dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
            kernel0_1<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, bt, BT, pi, pj, c0);
          }
        }
      }
      else if (__c0Len % __side0LenMax)
      {
        if (__c0Len % __side0LenMax == 1)
        {
          const AN5D_TYPE __side0Len = 1;
          const AN5D_TYPE __side1Len = 128;
          const AN5D_TYPE __side2Len = 30;
          const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
          const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
          const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
          const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
          const AN5D_TYPE __blockSize = 1 * __side2LenOl;
          assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
          dim3 k0_dimBlock(__blockSize, 1, 1);
          dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
          kernel0_1<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, bt, BT, pi, pj, c0);
        }
        else if (__c0Len % __side0LenMax == 2)
        {
          const AN5D_TYPE __side0Len = 2;
          const AN5D_TYPE __side1Len = 128;
          const AN5D_TYPE __side2Len = 28;
          const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
          const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
          const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
          const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
          const AN5D_TYPE __blockSize = 1 * __side2LenOl;
          assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
          dim3 k0_dimBlock(__blockSize, 1, 1);
          dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
          kernel0_2<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, bt, BT, pi, pj, c0);
        }
        else if (__c0Len % __side0LenMax == 3)
        {
          const AN5D_TYPE __side0Len = 3;
          const AN5D_TYPE __side1Len = 128;
          const AN5D_TYPE __side2Len = 26;
          const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
          const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
          const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
          const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
          const AN5D_TYPE __blockSize = 1 * __side2LenOl;
          assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
          dim3 k0_dimBlock(__blockSize, 1, 1);
          dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
          kernel0_3<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, bt, BT, pi, pj, c0);
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
  }
  else {
//    for (int t = 0; t < timestep; t++)
//#pragma omp parallel for
  }

  return (((end_time != 0.0) ? end_time : sb_time()) - start_time);
}

void step_pass(int dimsize, SB_TYPE (*A)[dimsize][dimsize], int pi, int pj, int bt, int T)
{
}
