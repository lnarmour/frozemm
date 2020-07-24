#include <assert.h>
#include <stdio.h>
#include "box3d4r-16x16-1-256_kernel.hu"
#define BENCH_DIM 3
#define BENCH_FPP 1457
#define BENCH_RAD 4

#include "common.h"

double kernel_stencil(SB_TYPE *A1, int compsize, int timestep, bool scop)
{
  double start_time = sb_time(), end_time = 0.0;
  int dimsize = compsize + BENCH_RAD * 2;
  SB_TYPE (*A)[dimsize][dimsize][dimsize]
    = (SB_TYPE (*)[dimsize][dimsize][dimsize])A1;

  if (scop) {
    if (dimsize >= 9 && timestep >= 1) {
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

      double *dev_A;
      
      cudaCheckReturn(cudaMalloc((void **) &dev_A, (size_t)(2) * (size_t)(dimsize) * (size_t)(dimsize) * (size_t)(dimsize) * sizeof(double)));
      
{
      cudaCheckReturn(cudaMemcpy(dev_A, A, (size_t)(2) * (size_t)(dimsize) * (size_t)(dimsize) * (size_t)(dimsize) * sizeof(double), cudaMemcpyHostToDevice));
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
      const AN5D_TYPE __c1Len = (dimsize - 4 - 4);
      const AN5D_TYPE __c1Pad = (4);
      #define __c1 c1
      const AN5D_TYPE __c2Len = (dimsize - 4 - 4);
      const AN5D_TYPE __c2Pad = (4);
      #define __c2 c2
      const AN5D_TYPE __c3Len = (dimsize - 4 - 4);
      const AN5D_TYPE __c3Pad = (4);
      #define __c3 c3
      const AN5D_TYPE __halo1 = 4;
      const AN5D_TYPE __halo2 = 4;
      const AN5D_TYPE __halo3 = 4;
      AN5D_TYPE c0;
      AN5D_TYPE __side0LenMax;
      {
        const AN5D_TYPE __side0Len = 1;
        const AN5D_TYPE __side1Len = 256;
        const AN5D_TYPE __side2Len = 8;
        const AN5D_TYPE __side3Len = 8;
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
          kernel0_1<<<k0_dimGrid, k0_dimBlock>>> (dev_A, dimsize, timestep, c0);
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
      cudaCheckReturn(cudaMemcpy(A, dev_A, (size_t)(2) * (size_t)(dimsize) * (size_t)(dimsize) * (size_t)(dimsize) * sizeof(double), cudaMemcpyDeviceToHost));
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
            A[(t+1)%2][i][j][k] =
              -3.240f*A[t%2][i-4][j][k] +
              0.0010f*A[t%2][i-4][j-4][k-4] +
              0.0020f*A[t%2][i-4][j-4][k-3] +
              0.0030f*A[t%2][i-4][j-4][k-2] +
              0.0040f*A[t%2][i-4][j-4][k-1] +
              0.0050f*A[t%2][i-4][j-4][k] +
              0.0060f*A[t%2][i-4][j-4][k+1] +
              0.0070f*A[t%2][i-4][j-4][k+2] +
              0.0080f*A[t%2][i-4][j-4][k+3] +
              0.0090f*A[t%2][i-4][j-4][k+4] +
              0.0100f*A[t%2][i-4][j-3][k-4] +
              0.0110f*A[t%2][i-4][j-3][k-3] +
              0.0120f*A[t%2][i-4][j-3][k-2] +
              0.0130f*A[t%2][i-4][j-3][k-1] +
              0.0140f*A[t%2][i-4][j-3][k] +
              0.0150f*A[t%2][i-4][j-3][k+1] +
              0.0160f*A[t%2][i-4][j-3][k+2] +
              0.0170f*A[t%2][i-4][j-3][k+3] +
              0.0180f*A[t%2][i-4][j-3][k+4] +
              0.0190f*A[t%2][i-4][j-2][k-4] +
              0.0200f*A[t%2][i-4][j-2][k-3] +
              0.0210f*A[t%2][i-4][j-2][k-2] +
              0.0220f*A[t%2][i-4][j-2][k-1] +
              0.0230f*A[t%2][i-4][j-2][k] +
              0.0240f*A[t%2][i-4][j-2][k+1] +
              0.0250f*A[t%2][i-4][j-2][k+2] +
              0.0260f*A[t%2][i-4][j-2][k+3] +
              0.0270f*A[t%2][i-4][j-2][k+4] +
              0.0280f*A[t%2][i-4][j-1][k-4] +
              0.0290f*A[t%2][i-4][j-1][k-3] +
              0.0300f*A[t%2][i-4][j-1][k-2] +
              0.0310f*A[t%2][i-4][j-1][k-1] +
              0.0320f*A[t%2][i-4][j-1][k] +
              0.0330f*A[t%2][i-4][j-1][k+1] +
              0.0340f*A[t%2][i-4][j-1][k+2] +
              0.0350f*A[t%2][i-4][j-1][k+3] +
              0.0360f*A[t%2][i-4][j-1][k+4] +
              0.0370f*A[t%2][i-4][j][k-4] +
              0.0380f*A[t%2][i-4][j][k-3] +
              0.0390f*A[t%2][i-4][j][k-2] +
              0.0400f*A[t%2][i-4][j][k-1] +
              0.0410f*A[t%2][i-4][j][k+1] +
              0.0420f*A[t%2][i-4][j][k+2] +
              0.0430f*A[t%2][i-4][j][k+3] +
              0.0440f*A[t%2][i-4][j][k+4] +
              0.0450f*A[t%2][i-4][j+1][k-4] +
              0.0460f*A[t%2][i-4][j+1][k-3] +
              0.0470f*A[t%2][i-4][j+1][k-2] +
              0.0480f*A[t%2][i-4][j+1][k-1] +
              0.0490f*A[t%2][i-4][j+1][k] +
              0.0500f*A[t%2][i-4][j+1][k+1] +
              0.0510f*A[t%2][i-4][j+1][k+2] +
              0.0520f*A[t%2][i-4][j+1][k+3] +
              0.0530f*A[t%2][i-4][j+1][k+4] +
              0.0540f*A[t%2][i-4][j+2][k-4] +
              0.0550f*A[t%2][i-4][j+2][k-3] +
              0.0560f*A[t%2][i-4][j+2][k-2] +
              0.0570f*A[t%2][i-4][j+2][k-1] +
              0.0580f*A[t%2][i-4][j+2][k] +
              0.0590f*A[t%2][i-4][j+2][k+1] +
              0.0600f*A[t%2][i-4][j+2][k+2] +
              0.0610f*A[t%2][i-4][j+2][k+3] +
              0.0620f*A[t%2][i-4][j+2][k+4] +
              0.0630f*A[t%2][i-4][j+3][k-4] +
              0.0640f*A[t%2][i-4][j+3][k-3] +
              0.0650f*A[t%2][i-4][j+3][k-2] +
              0.0660f*A[t%2][i-4][j+3][k-1] +
              0.0670f*A[t%2][i-4][j+3][k] +
              0.0680f*A[t%2][i-4][j+3][k+1] +
              0.0690f*A[t%2][i-4][j+3][k+2] +
              0.0700f*A[t%2][i-4][j+3][k+3] +
              0.0710f*A[t%2][i-4][j+3][k+4] +
              0.0720f*A[t%2][i-4][j+4][k-4] +
              0.0730f*A[t%2][i-4][j+4][k-3] +
              0.0740f*A[t%2][i-4][j+4][k-2] +
              0.0750f*A[t%2][i-4][j+4][k-1] +
              0.0760f*A[t%2][i-4][j+4][k] +
              0.0770f*A[t%2][i-4][j+4][k+1] +
              0.0780f*A[t%2][i-4][j+4][k+2] +
              0.0790f*A[t%2][i-4][j+4][k+3] +
              0.0800f*A[t%2][i-4][j+4][k+4] +

              -3.248f*A[t%2][i-3][j][k] +
              0.0011f*A[t%2][i-3][j-4][k-4] +
              0.0021f*A[t%2][i-3][j-4][k-3] +
              0.0031f*A[t%2][i-3][j-4][k-2] +
              0.0041f*A[t%2][i-3][j-4][k-1] +
              0.0051f*A[t%2][i-3][j-4][k] +
              0.0061f*A[t%2][i-3][j-4][k+1] +
              0.0071f*A[t%2][i-3][j-4][k+2] +
              0.0081f*A[t%2][i-3][j-4][k+3] +
              0.0091f*A[t%2][i-3][j-4][k+4] +
              0.0101f*A[t%2][i-3][j-3][k-4] +
              0.0111f*A[t%2][i-3][j-3][k-3] +
              0.0121f*A[t%2][i-3][j-3][k-2] +
              0.0131f*A[t%2][i-3][j-3][k-1] +
              0.0141f*A[t%2][i-3][j-3][k] +
              0.0151f*A[t%2][i-3][j-3][k+1] +
              0.0161f*A[t%2][i-3][j-3][k+2] +
              0.0171f*A[t%2][i-3][j-3][k+3] +
              0.0181f*A[t%2][i-3][j-3][k+4] +
              0.0191f*A[t%2][i-3][j-2][k-4] +
              0.0201f*A[t%2][i-3][j-2][k-3] +
              0.0211f*A[t%2][i-3][j-2][k-2] +
              0.0221f*A[t%2][i-3][j-2][k-1] +
              0.0231f*A[t%2][i-3][j-2][k] +
              0.0241f*A[t%2][i-3][j-2][k+1] +
              0.0251f*A[t%2][i-3][j-2][k+2] +
              0.0261f*A[t%2][i-3][j-2][k+3] +
              0.0271f*A[t%2][i-3][j-2][k+4] +
              0.0281f*A[t%2][i-3][j-1][k-4] +
              0.0291f*A[t%2][i-3][j-1][k-3] +
              0.0301f*A[t%2][i-3][j-1][k-2] +
              0.0311f*A[t%2][i-3][j-1][k-1] +
              0.0321f*A[t%2][i-3][j-1][k] +
              0.0331f*A[t%2][i-3][j-1][k+1] +
              0.0341f*A[t%2][i-3][j-1][k+2] +
              0.0351f*A[t%2][i-3][j-1][k+3] +
              0.0361f*A[t%2][i-3][j-1][k+4] +
              0.0371f*A[t%2][i-3][j][k-4] +
              0.0381f*A[t%2][i-3][j][k-3] +
              0.0391f*A[t%2][i-3][j][k-2] +
              0.0401f*A[t%2][i-3][j][k-1] +
              0.0411f*A[t%2][i-3][j][k+1] +
              0.0421f*A[t%2][i-3][j][k+2] +
              0.0431f*A[t%2][i-3][j][k+3] +
              0.0441f*A[t%2][i-3][j][k+4] +
              0.0451f*A[t%2][i-3][j+1][k-4] +
              0.0461f*A[t%2][i-3][j+1][k-3] +
              0.0471f*A[t%2][i-3][j+1][k-2] +
              0.0481f*A[t%2][i-3][j+1][k-1] +
              0.0491f*A[t%2][i-3][j+1][k] +
              0.0501f*A[t%2][i-3][j+1][k+1] +
              0.0511f*A[t%2][i-3][j+1][k+2] +
              0.0521f*A[t%2][i-3][j+1][k+3] +
              0.0531f*A[t%2][i-3][j+1][k+4] +
              0.0541f*A[t%2][i-3][j+2][k-4] +
              0.0551f*A[t%2][i-3][j+2][k-3] +
              0.0561f*A[t%2][i-3][j+2][k-2] +
              0.0571f*A[t%2][i-3][j+2][k-1] +
              0.0581f*A[t%2][i-3][j+2][k] +
              0.0591f*A[t%2][i-3][j+2][k+1] +
              0.0601f*A[t%2][i-3][j+2][k+2] +
              0.0611f*A[t%2][i-3][j+2][k+3] +
              0.0621f*A[t%2][i-3][j+2][k+4] +
              0.0631f*A[t%2][i-3][j+3][k-4] +
              0.0641f*A[t%2][i-3][j+3][k-3] +
              0.0651f*A[t%2][i-3][j+3][k-2] +
              0.0661f*A[t%2][i-3][j+3][k-1] +
              0.0671f*A[t%2][i-3][j+3][k] +
              0.0681f*A[t%2][i-3][j+3][k+1] +
              0.0691f*A[t%2][i-3][j+3][k+2] +
              0.0701f*A[t%2][i-3][j+3][k+3] +
              0.0711f*A[t%2][i-3][j+3][k+4] +
              0.0721f*A[t%2][i-3][j+4][k-4] +
              0.0731f*A[t%2][i-3][j+4][k-3] +
              0.0741f*A[t%2][i-3][j+4][k-2] +
              0.0751f*A[t%2][i-3][j+4][k-1] +
              0.0761f*A[t%2][i-3][j+4][k] +
              0.0771f*A[t%2][i-3][j+4][k+1] +
              0.0781f*A[t%2][i-3][j+4][k+2] +
              0.0791f*A[t%2][i-3][j+4][k+3] +
              0.0801f*A[t%2][i-3][j+4][k+4] +

              -3.256f*A[t%2][i-2][j][k] +
              0.0012f*A[t%2][i-2][j-4][k-4] +
              0.0022f*A[t%2][i-2][j-4][k-3] +
              0.0032f*A[t%2][i-2][j-4][k-2] +
              0.0042f*A[t%2][i-2][j-4][k-1] +
              0.0052f*A[t%2][i-2][j-4][k] +
              0.0062f*A[t%2][i-2][j-4][k+1] +
              0.0072f*A[t%2][i-2][j-4][k+2] +
              0.0082f*A[t%2][i-2][j-4][k+3] +
              0.0092f*A[t%2][i-2][j-4][k+4] +
              0.0102f*A[t%2][i-2][j-3][k-4] +
              0.0112f*A[t%2][i-2][j-3][k-3] +
              0.0122f*A[t%2][i-2][j-3][k-2] +
              0.0132f*A[t%2][i-2][j-3][k-1] +
              0.0142f*A[t%2][i-2][j-3][k] +
              0.0152f*A[t%2][i-2][j-3][k+1] +
              0.0162f*A[t%2][i-2][j-3][k+2] +
              0.0172f*A[t%2][i-2][j-3][k+3] +
              0.0182f*A[t%2][i-2][j-3][k+4] +
              0.0192f*A[t%2][i-2][j-2][k-4] +
              0.0202f*A[t%2][i-2][j-2][k-3] +
              0.0212f*A[t%2][i-2][j-2][k-2] +
              0.0222f*A[t%2][i-2][j-2][k-1] +
              0.0232f*A[t%2][i-2][j-2][k] +
              0.0242f*A[t%2][i-2][j-2][k+1] +
              0.0252f*A[t%2][i-2][j-2][k+2] +
              0.0262f*A[t%2][i-2][j-2][k+3] +
              0.0272f*A[t%2][i-2][j-2][k+4] +
              0.0282f*A[t%2][i-2][j-1][k-4] +
              0.0292f*A[t%2][i-2][j-1][k-3] +
              0.0302f*A[t%2][i-2][j-1][k-2] +
              0.0312f*A[t%2][i-2][j-1][k-1] +
              0.0322f*A[t%2][i-2][j-1][k] +
              0.0332f*A[t%2][i-2][j-1][k+1] +
              0.0342f*A[t%2][i-2][j-1][k+2] +
              0.0352f*A[t%2][i-2][j-1][k+3] +
              0.0362f*A[t%2][i-2][j-1][k+4] +
              0.0372f*A[t%2][i-2][j][k-4] +
              0.0382f*A[t%2][i-2][j][k-3] +
              0.0392f*A[t%2][i-2][j][k-2] +
              0.0402f*A[t%2][i-2][j][k-1] +
              0.0412f*A[t%2][i-2][j][k+1] +
              0.0422f*A[t%2][i-2][j][k+2] +
              0.0432f*A[t%2][i-2][j][k+3] +
              0.0442f*A[t%2][i-2][j][k+4] +
              0.0452f*A[t%2][i-2][j+1][k-4] +
              0.0462f*A[t%2][i-2][j+1][k-3] +
              0.0472f*A[t%2][i-2][j+1][k-2] +
              0.0482f*A[t%2][i-2][j+1][k-1] +
              0.0492f*A[t%2][i-2][j+1][k] +
              0.0502f*A[t%2][i-2][j+1][k+1] +
              0.0512f*A[t%2][i-2][j+1][k+2] +
              0.0522f*A[t%2][i-2][j+1][k+3] +
              0.0532f*A[t%2][i-2][j+1][k+4] +
              0.0542f*A[t%2][i-2][j+2][k-4] +
              0.0552f*A[t%2][i-2][j+2][k-3] +
              0.0562f*A[t%2][i-2][j+2][k-2] +
              0.0572f*A[t%2][i-2][j+2][k-1] +
              0.0582f*A[t%2][i-2][j+2][k] +
              0.0592f*A[t%2][i-2][j+2][k+1] +
              0.0602f*A[t%2][i-2][j+2][k+2] +
              0.0612f*A[t%2][i-2][j+2][k+3] +
              0.0622f*A[t%2][i-2][j+2][k+4] +
              0.0632f*A[t%2][i-2][j+3][k-4] +
              0.0642f*A[t%2][i-2][j+3][k-3] +
              0.0652f*A[t%2][i-2][j+3][k-2] +
              0.0662f*A[t%2][i-2][j+3][k-1] +
              0.0672f*A[t%2][i-2][j+3][k] +
              0.0682f*A[t%2][i-2][j+3][k+1] +
              0.0692f*A[t%2][i-2][j+3][k+2] +
              0.0702f*A[t%2][i-2][j+3][k+3] +
              0.0712f*A[t%2][i-2][j+3][k+4] +
              0.0722f*A[t%2][i-2][j+4][k-4] +
              0.0732f*A[t%2][i-2][j+4][k-3] +
              0.0742f*A[t%2][i-2][j+4][k-2] +
              0.0752f*A[t%2][i-2][j+4][k-1] +
              0.0762f*A[t%2][i-2][j+4][k] +
              0.0772f*A[t%2][i-2][j+4][k+1] +
              0.0782f*A[t%2][i-2][j+4][k+2] +
              0.0792f*A[t%2][i-2][j+4][k+3] +
              0.0802f*A[t%2][i-2][j+4][k+4] +

              -3.264f*A[t%2][i-1][j][k] +
              0.0013f*A[t%2][i-1][j-4][k-4] +
              0.0023f*A[t%2][i-1][j-4][k-3] +
              0.0033f*A[t%2][i-1][j-4][k-2] +
              0.0043f*A[t%2][i-1][j-4][k-1] +
              0.0053f*A[t%2][i-1][j-4][k] +
              0.0063f*A[t%2][i-1][j-4][k+1] +
              0.0073f*A[t%2][i-1][j-4][k+2] +
              0.0083f*A[t%2][i-1][j-4][k+3] +
              0.0093f*A[t%2][i-1][j-4][k+4] +
              0.0103f*A[t%2][i-1][j-3][k-4] +
              0.0113f*A[t%2][i-1][j-3][k-3] +
              0.0123f*A[t%2][i-1][j-3][k-2] +
              0.0133f*A[t%2][i-1][j-3][k-1] +
              0.0143f*A[t%2][i-1][j-3][k] +
              0.0153f*A[t%2][i-1][j-3][k+1] +
              0.0163f*A[t%2][i-1][j-3][k+2] +
              0.0173f*A[t%2][i-1][j-3][k+3] +
              0.0183f*A[t%2][i-1][j-3][k+4] +
              0.0193f*A[t%2][i-1][j-2][k-4] +
              0.0203f*A[t%2][i-1][j-2][k-3] +
              0.0213f*A[t%2][i-1][j-2][k-2] +
              0.0223f*A[t%2][i-1][j-2][k-1] +
              0.0233f*A[t%2][i-1][j-2][k] +
              0.0243f*A[t%2][i-1][j-2][k+1] +
              0.0253f*A[t%2][i-1][j-2][k+2] +
              0.0263f*A[t%2][i-1][j-2][k+3] +
              0.0273f*A[t%2][i-1][j-2][k+4] +
              0.0283f*A[t%2][i-1][j-1][k-4] +
              0.0293f*A[t%2][i-1][j-1][k-3] +
              0.0303f*A[t%2][i-1][j-1][k-2] +
              0.0313f*A[t%2][i-1][j-1][k-1] +
              0.0323f*A[t%2][i-1][j-1][k] +
              0.0333f*A[t%2][i-1][j-1][k+1] +
              0.0343f*A[t%2][i-1][j-1][k+2] +
              0.0353f*A[t%2][i-1][j-1][k+3] +
              0.0363f*A[t%2][i-1][j-1][k+4] +
              0.0373f*A[t%2][i-1][j][k-4] +
              0.0383f*A[t%2][i-1][j][k-3] +
              0.0393f*A[t%2][i-1][j][k-2] +
              0.0403f*A[t%2][i-1][j][k-1] +
              0.0413f*A[t%2][i-1][j][k+1] +
              0.0423f*A[t%2][i-1][j][k+2] +
              0.0433f*A[t%2][i-1][j][k+3] +
              0.0443f*A[t%2][i-1][j][k+4] +
              0.0453f*A[t%2][i-1][j+1][k-4] +
              0.0463f*A[t%2][i-1][j+1][k-3] +
              0.0473f*A[t%2][i-1][j+1][k-2] +
              0.0483f*A[t%2][i-1][j+1][k-1] +
              0.0493f*A[t%2][i-1][j+1][k] +
              0.0503f*A[t%2][i-1][j+1][k+1] +
              0.0513f*A[t%2][i-1][j+1][k+2] +
              0.0523f*A[t%2][i-1][j+1][k+3] +
              0.0533f*A[t%2][i-1][j+1][k+4] +
              0.0543f*A[t%2][i-1][j+2][k-4] +
              0.0553f*A[t%2][i-1][j+2][k-3] +
              0.0563f*A[t%2][i-1][j+2][k-2] +
              0.0573f*A[t%2][i-1][j+2][k-1] +
              0.0583f*A[t%2][i-1][j+2][k] +
              0.0593f*A[t%2][i-1][j+2][k+1] +
              0.0603f*A[t%2][i-1][j+2][k+2] +
              0.0613f*A[t%2][i-1][j+2][k+3] +
              0.0623f*A[t%2][i-1][j+2][k+4] +
              0.0633f*A[t%2][i-1][j+3][k-4] +
              0.0643f*A[t%2][i-1][j+3][k-3] +
              0.0653f*A[t%2][i-1][j+3][k-2] +
              0.0663f*A[t%2][i-1][j+3][k-1] +
              0.0673f*A[t%2][i-1][j+3][k] +
              0.0683f*A[t%2][i-1][j+3][k+1] +
              0.0693f*A[t%2][i-1][j+3][k+2] +
              0.0703f*A[t%2][i-1][j+3][k+3] +
              0.0713f*A[t%2][i-1][j+3][k+4] +
              0.0723f*A[t%2][i-1][j+4][k-4] +
              0.0733f*A[t%2][i-1][j+4][k-3] +
              0.0743f*A[t%2][i-1][j+4][k-2] +
              0.0753f*A[t%2][i-1][j+4][k-1] +
              0.0763f*A[t%2][i-1][j+4][k] +
              0.0773f*A[t%2][i-1][j+4][k+1] +
              0.0783f*A[t%2][i-1][j+4][k+2] +
              0.0793f*A[t%2][i-1][j+4][k+3] +
              0.0803f*A[t%2][i-1][j+4][k+4] +

              -3.272f*A[t%2][i][j][k] +
              0.0014f*A[t%2][i][j-4][k-4] +
              0.0024f*A[t%2][i][j-4][k-3] +
              0.0034f*A[t%2][i][j-4][k-2] +
              0.0044f*A[t%2][i][j-4][k-1] +
              0.0054f*A[t%2][i][j-4][k] +
              0.0064f*A[t%2][i][j-4][k+1] +
              0.0074f*A[t%2][i][j-4][k+2] +
              0.0084f*A[t%2][i][j-4][k+3] +
              0.0094f*A[t%2][i][j-4][k+4] +
              0.0104f*A[t%2][i][j-3][k-4] +
              0.0114f*A[t%2][i][j-3][k-3] +
              0.0124f*A[t%2][i][j-3][k-2] +
              0.0134f*A[t%2][i][j-3][k-1] +
              0.0144f*A[t%2][i][j-3][k] +
              0.0154f*A[t%2][i][j-3][k+1] +
              0.0164f*A[t%2][i][j-3][k+2] +
              0.0174f*A[t%2][i][j-3][k+3] +
              0.0184f*A[t%2][i][j-3][k+4] +
              0.0194f*A[t%2][i][j-2][k-4] +
              0.0204f*A[t%2][i][j-2][k-3] +
              0.0214f*A[t%2][i][j-2][k-2] +
              0.0224f*A[t%2][i][j-2][k-1] +
              0.0234f*A[t%2][i][j-2][k] +
              0.0244f*A[t%2][i][j-2][k+1] +
              0.0254f*A[t%2][i][j-2][k+2] +
              0.0264f*A[t%2][i][j-2][k+3] +
              0.0274f*A[t%2][i][j-2][k+4] +
              0.0284f*A[t%2][i][j-1][k-4] +
              0.0294f*A[t%2][i][j-1][k-3] +
              0.0304f*A[t%2][i][j-1][k-2] +
              0.0314f*A[t%2][i][j-1][k-1] +
              0.0324f*A[t%2][i][j-1][k] +
              0.0334f*A[t%2][i][j-1][k+1] +
              0.0344f*A[t%2][i][j-1][k+2] +
              0.0354f*A[t%2][i][j-1][k+3] +
              0.0364f*A[t%2][i][j-1][k+4] +
              0.0374f*A[t%2][i][j][k-4] +
              0.0384f*A[t%2][i][j][k-3] +
              0.0394f*A[t%2][i][j][k-2] +
              0.0404f*A[t%2][i][j][k-1] +
              0.0414f*A[t%2][i][j][k+1] +
              0.0424f*A[t%2][i][j][k+2] +
              0.0434f*A[t%2][i][j][k+3] +
              0.0444f*A[t%2][i][j][k+4] +
              0.0454f*A[t%2][i][j+1][k-4] +
              0.0464f*A[t%2][i][j+1][k-3] +
              0.0474f*A[t%2][i][j+1][k-2] +
              0.0484f*A[t%2][i][j+1][k-1] +
              0.0494f*A[t%2][i][j+1][k] +
              0.0504f*A[t%2][i][j+1][k+1] +
              0.0514f*A[t%2][i][j+1][k+2] +
              0.0524f*A[t%2][i][j+1][k+3] +
              0.0534f*A[t%2][i][j+1][k+4] +
              0.0544f*A[t%2][i][j+2][k-4] +
              0.0554f*A[t%2][i][j+2][k-3] +
              0.0564f*A[t%2][i][j+2][k-2] +
              0.0574f*A[t%2][i][j+2][k-1] +
              0.0584f*A[t%2][i][j+2][k] +
              0.0594f*A[t%2][i][j+2][k+1] +
              0.0604f*A[t%2][i][j+2][k+2] +
              0.0614f*A[t%2][i][j+2][k+3] +
              0.0624f*A[t%2][i][j+2][k+4] +
              0.0634f*A[t%2][i][j+3][k-4] +
              0.0644f*A[t%2][i][j+3][k-3] +
              0.0654f*A[t%2][i][j+3][k-2] +
              0.0664f*A[t%2][i][j+3][k-1] +
              0.0674f*A[t%2][i][j+3][k] +
              0.0684f*A[t%2][i][j+3][k+1] +
              0.0694f*A[t%2][i][j+3][k+2] +
              0.0704f*A[t%2][i][j+3][k+3] +
              0.0714f*A[t%2][i][j+3][k+4] +
              0.0724f*A[t%2][i][j+4][k-4] +
              0.0734f*A[t%2][i][j+4][k-3] +
              0.0744f*A[t%2][i][j+4][k-2] +
              0.0754f*A[t%2][i][j+4][k-1] +
              0.0764f*A[t%2][i][j+4][k] +
              0.0774f*A[t%2][i][j+4][k+1] +
              0.0784f*A[t%2][i][j+4][k+2] +
              0.0794f*A[t%2][i][j+4][k+3] +
              0.0804f*A[t%2][i][j+4][k+4] +

              -3.280f*A[t%2][i+1][j][k] +
              0.0015f*A[t%2][i+1][j-4][k-4] +
              0.0025f*A[t%2][i+1][j-4][k-3] +
              0.0035f*A[t%2][i+1][j-4][k-2] +
              0.0045f*A[t%2][i+1][j-4][k-1] +
              0.0055f*A[t%2][i+1][j-4][k] +
              0.0065f*A[t%2][i+1][j-4][k+1] +
              0.0075f*A[t%2][i+1][j-4][k+2] +
              0.0085f*A[t%2][i+1][j-4][k+3] +
              0.0095f*A[t%2][i+1][j-4][k+4] +
              0.0105f*A[t%2][i+1][j-3][k-4] +
              0.0115f*A[t%2][i+1][j-3][k-3] +
              0.0125f*A[t%2][i+1][j-3][k-2] +
              0.0135f*A[t%2][i+1][j-3][k-1] +
              0.0145f*A[t%2][i+1][j-3][k] +
              0.0155f*A[t%2][i+1][j-3][k+1] +
              0.0165f*A[t%2][i+1][j-3][k+2] +
              0.0175f*A[t%2][i+1][j-3][k+3] +
              0.0185f*A[t%2][i+1][j-3][k+4] +
              0.0195f*A[t%2][i+1][j-2][k-4] +
              0.0205f*A[t%2][i+1][j-2][k-3] +
              0.0215f*A[t%2][i+1][j-2][k-2] +
              0.0225f*A[t%2][i+1][j-2][k-1] +
              0.0235f*A[t%2][i+1][j-2][k] +
              0.0245f*A[t%2][i+1][j-2][k+1] +
              0.0255f*A[t%2][i+1][j-2][k+2] +
              0.0265f*A[t%2][i+1][j-2][k+3] +
              0.0275f*A[t%2][i+1][j-2][k+4] +
              0.0285f*A[t%2][i+1][j-1][k-4] +
              0.0295f*A[t%2][i+1][j-1][k-3] +
              0.0305f*A[t%2][i+1][j-1][k-2] +
              0.0315f*A[t%2][i+1][j-1][k-1] +
              0.0325f*A[t%2][i+1][j-1][k] +
              0.0335f*A[t%2][i+1][j-1][k+1] +
              0.0345f*A[t%2][i+1][j-1][k+2] +
              0.0355f*A[t%2][i+1][j-1][k+3] +
              0.0365f*A[t%2][i+1][j-1][k+4] +
              0.0375f*A[t%2][i+1][j][k-4] +
              0.0385f*A[t%2][i+1][j][k-3] +
              0.0395f*A[t%2][i+1][j][k-2] +
              0.0405f*A[t%2][i+1][j][k-1] +
              0.0415f*A[t%2][i+1][j][k+1] +
              0.0425f*A[t%2][i+1][j][k+2] +
              0.0435f*A[t%2][i+1][j][k+3] +
              0.0445f*A[t%2][i+1][j][k+4] +
              0.0455f*A[t%2][i+1][j+1][k-4] +
              0.0465f*A[t%2][i+1][j+1][k-3] +
              0.0475f*A[t%2][i+1][j+1][k-2] +
              0.0485f*A[t%2][i+1][j+1][k-1] +
              0.0495f*A[t%2][i+1][j+1][k] +
              0.0505f*A[t%2][i+1][j+1][k+1] +
              0.0515f*A[t%2][i+1][j+1][k+2] +
              0.0525f*A[t%2][i+1][j+1][k+3] +
              0.0535f*A[t%2][i+1][j+1][k+4] +
              0.0545f*A[t%2][i+1][j+2][k-4] +
              0.0555f*A[t%2][i+1][j+2][k-3] +
              0.0565f*A[t%2][i+1][j+2][k-2] +
              0.0575f*A[t%2][i+1][j+2][k-1] +
              0.0585f*A[t%2][i+1][j+2][k] +
              0.0595f*A[t%2][i+1][j+2][k+1] +
              0.0605f*A[t%2][i+1][j+2][k+2] +
              0.0615f*A[t%2][i+1][j+2][k+3] +
              0.0625f*A[t%2][i+1][j+2][k+4] +
              0.0635f*A[t%2][i+1][j+3][k-4] +
              0.0645f*A[t%2][i+1][j+3][k-3] +
              0.0655f*A[t%2][i+1][j+3][k-2] +
              0.0665f*A[t%2][i+1][j+3][k-1] +
              0.0675f*A[t%2][i+1][j+3][k] +
              0.0685f*A[t%2][i+1][j+3][k+1] +
              0.0695f*A[t%2][i+1][j+3][k+2] +
              0.0705f*A[t%2][i+1][j+3][k+3] +
              0.0715f*A[t%2][i+1][j+3][k+4] +
              0.0725f*A[t%2][i+1][j+4][k-4] +
              0.0735f*A[t%2][i+1][j+4][k-3] +
              0.0745f*A[t%2][i+1][j+4][k-2] +
              0.0755f*A[t%2][i+1][j+4][k-1] +
              0.0765f*A[t%2][i+1][j+4][k] +
              0.0775f*A[t%2][i+1][j+4][k+1] +
              0.0785f*A[t%2][i+1][j+4][k+2] +
              0.0795f*A[t%2][i+1][j+4][k+3] +
              0.0805f*A[t%2][i+1][j+4][k+4] +

              -3.288f*A[t%2][i+2][j][k] +
              0.0016f*A[t%2][i+2][j-4][k-4] +
              0.0026f*A[t%2][i+2][j-4][k-3] +
              0.0036f*A[t%2][i+2][j-4][k-2] +
              0.0046f*A[t%2][i+2][j-4][k-1] +
              0.0056f*A[t%2][i+2][j-4][k] +
              0.0066f*A[t%2][i+2][j-4][k+1] +
              0.0076f*A[t%2][i+2][j-4][k+2] +
              0.0086f*A[t%2][i+2][j-4][k+3] +
              0.0096f*A[t%2][i+2][j-4][k+4] +
              0.0106f*A[t%2][i+2][j-3][k-4] +
              0.0116f*A[t%2][i+2][j-3][k-3] +
              0.0126f*A[t%2][i+2][j-3][k-2] +
              0.0136f*A[t%2][i+2][j-3][k-1] +
              0.0146f*A[t%2][i+2][j-3][k] +
              0.0156f*A[t%2][i+2][j-3][k+1] +
              0.0166f*A[t%2][i+2][j-3][k+2] +
              0.0176f*A[t%2][i+2][j-3][k+3] +
              0.0186f*A[t%2][i+2][j-3][k+4] +
              0.0196f*A[t%2][i+2][j-2][k-4] +
              0.0206f*A[t%2][i+2][j-2][k-3] +
              0.0216f*A[t%2][i+2][j-2][k-2] +
              0.0226f*A[t%2][i+2][j-2][k-1] +
              0.0236f*A[t%2][i+2][j-2][k] +
              0.0246f*A[t%2][i+2][j-2][k+1] +
              0.0256f*A[t%2][i+2][j-2][k+2] +
              0.0266f*A[t%2][i+2][j-2][k+3] +
              0.0276f*A[t%2][i+2][j-2][k+4] +
              0.0286f*A[t%2][i+2][j-1][k-4] +
              0.0296f*A[t%2][i+2][j-1][k-3] +
              0.0306f*A[t%2][i+2][j-1][k-2] +
              0.0316f*A[t%2][i+2][j-1][k-1] +
              0.0326f*A[t%2][i+2][j-1][k] +
              0.0336f*A[t%2][i+2][j-1][k+1] +
              0.0346f*A[t%2][i+2][j-1][k+2] +
              0.0356f*A[t%2][i+2][j-1][k+3] +
              0.0366f*A[t%2][i+2][j-1][k+4] +
              0.0376f*A[t%2][i+2][j][k-4] +
              0.0386f*A[t%2][i+2][j][k-3] +
              0.0396f*A[t%2][i+2][j][k-2] +
              0.0406f*A[t%2][i+2][j][k-1] +
              0.0416f*A[t%2][i+2][j][k+1] +
              0.0426f*A[t%2][i+2][j][k+2] +
              0.0436f*A[t%2][i+2][j][k+3] +
              0.0446f*A[t%2][i+2][j][k+4] +
              0.0456f*A[t%2][i+2][j+1][k-4] +
              0.0466f*A[t%2][i+2][j+1][k-3] +
              0.0476f*A[t%2][i+2][j+1][k-2] +
              0.0486f*A[t%2][i+2][j+1][k-1] +
              0.0496f*A[t%2][i+2][j+1][k] +
              0.0506f*A[t%2][i+2][j+1][k+1] +
              0.0516f*A[t%2][i+2][j+1][k+2] +
              0.0526f*A[t%2][i+2][j+1][k+3] +
              0.0536f*A[t%2][i+2][j+1][k+4] +
              0.0546f*A[t%2][i+2][j+2][k-4] +
              0.0556f*A[t%2][i+2][j+2][k-3] +
              0.0566f*A[t%2][i+2][j+2][k-2] +
              0.0576f*A[t%2][i+2][j+2][k-1] +
              0.0586f*A[t%2][i+2][j+2][k] +
              0.0596f*A[t%2][i+2][j+2][k+1] +
              0.0606f*A[t%2][i+2][j+2][k+2] +
              0.0616f*A[t%2][i+2][j+2][k+3] +
              0.0626f*A[t%2][i+2][j+2][k+4] +
              0.0636f*A[t%2][i+2][j+3][k-4] +
              0.0646f*A[t%2][i+2][j+3][k-3] +
              0.0656f*A[t%2][i+2][j+3][k-2] +
              0.0666f*A[t%2][i+2][j+3][k-1] +
              0.0676f*A[t%2][i+2][j+3][k] +
              0.0686f*A[t%2][i+2][j+3][k+1] +
              0.0696f*A[t%2][i+2][j+3][k+2] +
              0.0706f*A[t%2][i+2][j+3][k+3] +
              0.0716f*A[t%2][i+2][j+3][k+4] +
              0.0726f*A[t%2][i+2][j+4][k-4] +
              0.0736f*A[t%2][i+2][j+4][k-3] +
              0.0746f*A[t%2][i+2][j+4][k-2] +
              0.0756f*A[t%2][i+2][j+4][k-1] +
              0.0766f*A[t%2][i+2][j+4][k] +
              0.0776f*A[t%2][i+2][j+4][k+1] +
              0.0786f*A[t%2][i+2][j+4][k+2] +
              0.0796f*A[t%2][i+2][j+4][k+3] +
              0.0806f*A[t%2][i+2][j+4][k+4] +

              -3.296f*A[t%2][i+3][j][k] +
              0.0017f*A[t%2][i+3][j-4][k-4] +
              0.0027f*A[t%2][i+3][j-4][k-3] +
              0.0037f*A[t%2][i+3][j-4][k-2] +
              0.0047f*A[t%2][i+3][j-4][k-1] +
              0.0057f*A[t%2][i+3][j-4][k] +
              0.0067f*A[t%2][i+3][j-4][k+1] +
              0.0077f*A[t%2][i+3][j-4][k+2] +
              0.0087f*A[t%2][i+3][j-4][k+3] +
              0.0097f*A[t%2][i+3][j-4][k+4] +
              0.0107f*A[t%2][i+3][j-3][k-4] +
              0.0117f*A[t%2][i+3][j-3][k-3] +
              0.0127f*A[t%2][i+3][j-3][k-2] +
              0.0137f*A[t%2][i+3][j-3][k-1] +
              0.0147f*A[t%2][i+3][j-3][k] +
              0.0157f*A[t%2][i+3][j-3][k+1] +
              0.0167f*A[t%2][i+3][j-3][k+2] +
              0.0177f*A[t%2][i+3][j-3][k+3] +
              0.0187f*A[t%2][i+3][j-3][k+4] +
              0.0197f*A[t%2][i+3][j-2][k-4] +
              0.0207f*A[t%2][i+3][j-2][k-3] +
              0.0217f*A[t%2][i+3][j-2][k-2] +
              0.0227f*A[t%2][i+3][j-2][k-1] +
              0.0237f*A[t%2][i+3][j-2][k] +
              0.0247f*A[t%2][i+3][j-2][k+1] +
              0.0257f*A[t%2][i+3][j-2][k+2] +
              0.0267f*A[t%2][i+3][j-2][k+3] +
              0.0277f*A[t%2][i+3][j-2][k+4] +
              0.0287f*A[t%2][i+3][j-1][k-4] +
              0.0297f*A[t%2][i+3][j-1][k-3] +
              0.0307f*A[t%2][i+3][j-1][k-2] +
              0.0317f*A[t%2][i+3][j-1][k-1] +
              0.0327f*A[t%2][i+3][j-1][k] +
              0.0337f*A[t%2][i+3][j-1][k+1] +
              0.0347f*A[t%2][i+3][j-1][k+2] +
              0.0357f*A[t%2][i+3][j-1][k+3] +
              0.0367f*A[t%2][i+3][j-1][k+4] +
              0.0377f*A[t%2][i+3][j][k-4] +
              0.0387f*A[t%2][i+3][j][k-3] +
              0.0397f*A[t%2][i+3][j][k-2] +
              0.0407f*A[t%2][i+3][j][k-1] +
              0.0417f*A[t%2][i+3][j][k+1] +
              0.0427f*A[t%2][i+3][j][k+2] +
              0.0437f*A[t%2][i+3][j][k+3] +
              0.0447f*A[t%2][i+3][j][k+4] +
              0.0457f*A[t%2][i+3][j+1][k-4] +
              0.0467f*A[t%2][i+3][j+1][k-3] +
              0.0477f*A[t%2][i+3][j+1][k-2] +
              0.0487f*A[t%2][i+3][j+1][k-1] +
              0.0497f*A[t%2][i+3][j+1][k] +
              0.0507f*A[t%2][i+3][j+1][k+1] +
              0.0517f*A[t%2][i+3][j+1][k+2] +
              0.0527f*A[t%2][i+3][j+1][k+3] +
              0.0537f*A[t%2][i+3][j+1][k+4] +
              0.0547f*A[t%2][i+3][j+2][k-4] +
              0.0557f*A[t%2][i+3][j+2][k-3] +
              0.0567f*A[t%2][i+3][j+2][k-2] +
              0.0577f*A[t%2][i+3][j+2][k-1] +
              0.0587f*A[t%2][i+3][j+2][k] +
              0.0597f*A[t%2][i+3][j+2][k+1] +
              0.0607f*A[t%2][i+3][j+2][k+2] +
              0.0617f*A[t%2][i+3][j+2][k+3] +
              0.0627f*A[t%2][i+3][j+2][k+4] +
              0.0637f*A[t%2][i+3][j+3][k-4] +
              0.0647f*A[t%2][i+3][j+3][k-3] +
              0.0657f*A[t%2][i+3][j+3][k-2] +
              0.0667f*A[t%2][i+3][j+3][k-1] +
              0.0677f*A[t%2][i+3][j+3][k] +
              0.0687f*A[t%2][i+3][j+3][k+1] +
              0.0697f*A[t%2][i+3][j+3][k+2] +
              0.0707f*A[t%2][i+3][j+3][k+3] +
              0.0717f*A[t%2][i+3][j+3][k+4] +
              0.0727f*A[t%2][i+3][j+4][k-4] +
              0.0737f*A[t%2][i+3][j+4][k-3] +
              0.0747f*A[t%2][i+3][j+4][k-2] +
              0.0757f*A[t%2][i+3][j+4][k-1] +
              0.0767f*A[t%2][i+3][j+4][k] +
              0.0777f*A[t%2][i+3][j+4][k+1] +
              0.0787f*A[t%2][i+3][j+4][k+2] +
              0.0797f*A[t%2][i+3][j+4][k+3] +
              0.0807f*A[t%2][i+3][j+4][k+4] +

              -3.304f*A[t%2][i+4][j][k] +
              0.0018f*A[t%2][i+4][j-4][k-4] +
              0.0028f*A[t%2][i+4][j-4][k-3] +
              0.0038f*A[t%2][i+4][j-4][k-2] +
              0.0048f*A[t%2][i+4][j-4][k-1] +
              0.0058f*A[t%2][i+4][j-4][k] +
              0.0068f*A[t%2][i+4][j-4][k+1] +
              0.0078f*A[t%2][i+4][j-4][k+2] +
              0.0088f*A[t%2][i+4][j-4][k+3] +
              0.0098f*A[t%2][i+4][j-4][k+4] +
              0.0108f*A[t%2][i+4][j-3][k-4] +
              0.0118f*A[t%2][i+4][j-3][k-3] +
              0.0128f*A[t%2][i+4][j-3][k-2] +
              0.0138f*A[t%2][i+4][j-3][k-1] +
              0.0148f*A[t%2][i+4][j-3][k] +
              0.0158f*A[t%2][i+4][j-3][k+1] +
              0.0168f*A[t%2][i+4][j-3][k+2] +
              0.0178f*A[t%2][i+4][j-3][k+3] +
              0.0188f*A[t%2][i+4][j-3][k+4] +
              0.0198f*A[t%2][i+4][j-2][k-4] +
              0.0208f*A[t%2][i+4][j-2][k-3] +
              0.0218f*A[t%2][i+4][j-2][k-2] +
              0.0228f*A[t%2][i+4][j-2][k-1] +
              0.0238f*A[t%2][i+4][j-2][k] +
              0.0248f*A[t%2][i+4][j-2][k+1] +
              0.0258f*A[t%2][i+4][j-2][k+2] +
              0.0268f*A[t%2][i+4][j-2][k+3] +
              0.0278f*A[t%2][i+4][j-2][k+4] +
              0.0288f*A[t%2][i+4][j-1][k-4] +
              0.0298f*A[t%2][i+4][j-1][k-3] +
              0.0308f*A[t%2][i+4][j-1][k-2] +
              0.0318f*A[t%2][i+4][j-1][k-1] +
              0.0328f*A[t%2][i+4][j-1][k] +
              0.0338f*A[t%2][i+4][j-1][k+1] +
              0.0348f*A[t%2][i+4][j-1][k+2] +
              0.0358f*A[t%2][i+4][j-1][k+3] +
              0.0368f*A[t%2][i+4][j-1][k+4] +
              0.0378f*A[t%2][i+4][j][k-4] +
              0.0388f*A[t%2][i+4][j][k-3] +
              0.0398f*A[t%2][i+4][j][k-2] +
              0.0408f*A[t%2][i+4][j][k-1] +
              0.0418f*A[t%2][i+4][j][k+1] +
              0.0428f*A[t%2][i+4][j][k+2] +
              0.0438f*A[t%2][i+4][j][k+3] +
              0.0448f*A[t%2][i+4][j][k+4] +
              0.0458f*A[t%2][i+4][j+1][k-4] +
              0.0468f*A[t%2][i+4][j+1][k-3] +
              0.0478f*A[t%2][i+4][j+1][k-2] +
              0.0488f*A[t%2][i+4][j+1][k-1] +
              0.0498f*A[t%2][i+4][j+1][k] +
              0.0508f*A[t%2][i+4][j+1][k+1] +
              0.0518f*A[t%2][i+4][j+1][k+2] +
              0.0528f*A[t%2][i+4][j+1][k+3] +
              0.0538f*A[t%2][i+4][j+1][k+4] +
              0.0548f*A[t%2][i+4][j+2][k-4] +
              0.0558f*A[t%2][i+4][j+2][k-3] +
              0.0568f*A[t%2][i+4][j+2][k-2] +
              0.0578f*A[t%2][i+4][j+2][k-1] +
              0.0588f*A[t%2][i+4][j+2][k] +
              0.0598f*A[t%2][i+4][j+2][k+1] +
              0.0608f*A[t%2][i+4][j+2][k+2] +
              0.0618f*A[t%2][i+4][j+2][k+3] +
              0.0628f*A[t%2][i+4][j+2][k+4] +
              0.0638f*A[t%2][i+4][j+3][k-4] +
              0.0648f*A[t%2][i+4][j+3][k-3] +
              0.0658f*A[t%2][i+4][j+3][k-2] +
              0.0668f*A[t%2][i+4][j+3][k-1] +
              0.0678f*A[t%2][i+4][j+3][k] +
              0.0688f*A[t%2][i+4][j+3][k+1] +
              0.0698f*A[t%2][i+4][j+3][k+2] +
              0.0708f*A[t%2][i+4][j+3][k+3] +
              0.0718f*A[t%2][i+4][j+3][k+4] +
              0.0728f*A[t%2][i+4][j+4][k-4] +
              0.0738f*A[t%2][i+4][j+4][k-3] +
              0.0748f*A[t%2][i+4][j+4][k-2] +
              0.0758f*A[t%2][i+4][j+4][k-1] +
              0.0768f*A[t%2][i+4][j+4][k] +
              0.0778f*A[t%2][i+4][j+4][k+1] +
              0.0788f*A[t%2][i+4][j+4][k+2] +
              0.0798f*A[t%2][i+4][j+4][k+3] +
              0.0808f*A[t%2][i+4][j+4][k+4];
  }

  return (((end_time != 0.0) ? end_time : sb_time()) - start_time);
}
