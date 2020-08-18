#include <assert.h>
#include <stdio.h>
#include "star2d1r-mp_kernel.hu"
#define BENCH_DIM 2
#define BENCH_FPP 9
#define BENCH_RAD 1

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

#define PI 896
#define PJ 896
#define BT 4

#include "common.h"

double
kernel_stencil(SB_TYPE * A1, int compsize, int T, bool scop)
{
	double		start_time = sb_time(), end_time = 0.0;
	int		dimsize = compsize + BENCH_RAD * 2;
	SB_TYPE(*A)[dimsize][dimsize] = (SB_TYPE(*)[dimsize][dimsize]) A1;

	int		t         , i, j;
	int		I = PI;
	int		J = PJ;

	if (scop) {
		{
			float *dev_A;
			float *dev_A_outer;

			cudaCheckReturn(cudaMalloc((void **)&dev_A_outer, (size_t) (2) * (size_t) (dimsize) * (size_t) (dimsize) * sizeof(float)));
			{
				cudaCheckReturn(cudaMemcpy(dev_A_outer, A, (size_t) (2) * (size_t) (dimsize) * (size_t) (dimsize) * sizeof(float), cudaMemcpyHostToDevice));
#ifdef STENCILBENCH
				cudaDeviceSynchronize();
				SB_START_INSTRUMENTS;
#endif
			}

      int pi = 2;
      int pj = 2;

      dev_A = &(dev_A_outer[pi*PI*dimsize + pj*PJ]);

		  //shift dev_A pointer and make multiple calls
			for (int bt = 0; bt < T; bt += BT) {
				timestep = BT;
				if (dimsize >= 2 && timestep >= 1 && I >= 3 && J >= 3) {
					{
						//START KERNEL
#ifndef AN5D_TYPE
#define AN5D_TYPE unsigned
#endif
							const AN5D_TYPE	__c0Len = (timestep - 0);
						const AN5D_TYPE	__c0Pad = (0);
#define __c0 c0
						const AN5D_TYPE	__c1Len = (min(dimsize, I - 1) - 1);
						const AN5D_TYPE	__c1Pad = (1);
#define __c1 c1
						const AN5D_TYPE	__c2Len = (min(dimsize, J - 1) - 1);
						const AN5D_TYPE	__c2Pad = (1);
#define __c2 c2
						const AN5D_TYPE	__halo1 = 1;
						const AN5D_TYPE	__halo2 = 1;
						AN5D_TYPE	c0;
						AN5D_TYPE	__side0LenMax;
						{
							const AN5D_TYPE	__side0Len = 4;
							const AN5D_TYPE	__side1Len = 128;
							const AN5D_TYPE	__side2Len = 24;
							const AN5D_TYPE	__OlLen1 = (__halo1 * __side0Len);
							const AN5D_TYPE	__OlLen2 = (__halo2 * __side0Len);
							const AN5D_TYPE	__side1LenOl = (__side1Len + 2 * __OlLen1);
							const AN5D_TYPE	__side2LenOl = (__side2Len + 2 * __OlLen2);
							const AN5D_TYPE	__blockSize = 1 * __side2LenOl;
							assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
							dim3		k0_dimBlock(__blockSize, 1, 1);
							dim3		k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
							AN5D_TYPE	__c0Padr = (__c0Len % 2) != (((__c0Len + __side0Len - 1) / __side0Len) % 2) && __c0Len % __side0Len < 2 ? 1 : 0;
							__side0LenMax = __side0Len;
							for (c0 = __c0Pad; c0 < __c0Pad + __c0Len / __side0Len - __c0Padr; c0 += 1) {
								kernel0_4 << <k0_dimGrid, k0_dimBlock >> >(dev_A, dimsize, timestep, I, J, c0);
							}
						}
						if ((__c0Len % 2) != (((__c0Len + __side0LenMax - 1) / __side0LenMax) % 2)) {
							if (__c0Len % __side0LenMax == 0) {
								{
									const AN5D_TYPE	__side0Len = 2;
									const AN5D_TYPE	__side1Len = 128;
									const AN5D_TYPE	__side2Len = 28;
									const AN5D_TYPE	__OlLen1 = (__halo1 * __side0Len);
									const AN5D_TYPE	__OlLen2 = (__halo2 * __side0Len);
									const AN5D_TYPE	__side1LenOl = (__side1Len + 2 * __OlLen1);
									const AN5D_TYPE	__side2LenOl = (__side2Len + 2 * __OlLen2);
									const AN5D_TYPE	__blockSize = 1 * __side2LenOl;
									assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
									dim3		k0_dimBlock(__blockSize, 1, 1);
									dim3		k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
									kernel0_2 << <k0_dimGrid, k0_dimBlock >> >(dev_A, dimsize, timestep, I, J, c0);
								}
								c0 += 1;
								{
									const AN5D_TYPE	__side0Len = 2;
									const AN5D_TYPE	__side1Len = 128;
									const AN5D_TYPE	__side2Len = 28;
									const AN5D_TYPE	__OlLen1 = (__halo1 * __side0Len);
									const AN5D_TYPE	__OlLen2 = (__halo2 * __side0Len);
									const AN5D_TYPE	__side1LenOl = (__side1Len + 2 * __OlLen1);
									const AN5D_TYPE	__side2LenOl = (__side2Len + 2 * __OlLen2);
									const AN5D_TYPE	__blockSize = 1 * __side2LenOl;
									assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
									dim3		k0_dimBlock(__blockSize, 1, 1);
									dim3		k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
									kernel0_2 << <k0_dimGrid, k0_dimBlock >> >(dev_A, dimsize, timestep, I, J, c0);
								}
							} else if (__c0Len % __side0LenMax == 1) {
								{
									const AN5D_TYPE	__side0Len = 3;
									const AN5D_TYPE	__side1Len = 128;
									const AN5D_TYPE	__side2Len = 26;
									const AN5D_TYPE	__OlLen1 = (__halo1 * __side0Len);
									const AN5D_TYPE	__OlLen2 = (__halo2 * __side0Len);
									const AN5D_TYPE	__side1LenOl = (__side1Len + 2 * __OlLen1);
									const AN5D_TYPE	__side2LenOl = (__side2Len + 2 * __OlLen2);
									const AN5D_TYPE	__blockSize = 1 * __side2LenOl;
									assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
									dim3		k0_dimBlock(__blockSize, 1, 1);
									dim3		k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
									kernel0_3 << <k0_dimGrid, k0_dimBlock >> >(dev_A, dimsize, timestep, I, J, c0);
								}
								c0 += 1;
								{
									const AN5D_TYPE	__side0Len = 1;
									const AN5D_TYPE	__side1Len = 128;
									const AN5D_TYPE	__side2Len = 30;
									const AN5D_TYPE	__OlLen1 = (__halo1 * __side0Len);
									const AN5D_TYPE	__OlLen2 = (__halo2 * __side0Len);
									const AN5D_TYPE	__side1LenOl = (__side1Len + 2 * __OlLen1);
									const AN5D_TYPE	__side2LenOl = (__side2Len + 2 * __OlLen2);
									const AN5D_TYPE	__blockSize = 1 * __side2LenOl;
									assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
									dim3		k0_dimBlock(__blockSize, 1, 1);
									dim3		k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
									kernel0_1 << <k0_dimGrid, k0_dimBlock >> >(dev_A, dimsize, timestep, I, J, c0);
								}
								c0 += 1;
								{
									const AN5D_TYPE	__side0Len = 1;
									const AN5D_TYPE	__side1Len = 128;
									const AN5D_TYPE	__side2Len = 30;
									const AN5D_TYPE	__OlLen1 = (__halo1 * __side0Len);
									const AN5D_TYPE	__OlLen2 = (__halo2 * __side0Len);
									const AN5D_TYPE	__side1LenOl = (__side1Len + 2 * __OlLen1);
									const AN5D_TYPE	__side2LenOl = (__side2Len + 2 * __OlLen2);
									const AN5D_TYPE	__blockSize = 1 * __side2LenOl;
									assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
									dim3		k0_dimBlock(__blockSize, 1, 1);
									dim3		k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
									kernel0_1 << <k0_dimGrid, k0_dimBlock >> >(dev_A, dimsize, timestep, I, J, c0);
								}
							} else if (__c0Len % __side0LenMax == 2) {
								{
									const AN5D_TYPE	__side0Len = 1;
									const AN5D_TYPE	__side1Len = 128;
									const AN5D_TYPE	__side2Len = 30;
									const AN5D_TYPE	__OlLen1 = (__halo1 * __side0Len);
									const AN5D_TYPE	__OlLen2 = (__halo2 * __side0Len);
									const AN5D_TYPE	__side1LenOl = (__side1Len + 2 * __OlLen1);
									const AN5D_TYPE	__side2LenOl = (__side2Len + 2 * __OlLen2);
									const AN5D_TYPE	__blockSize = 1 * __side2LenOl;
									assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
									dim3		k0_dimBlock(__blockSize, 1, 1);
									dim3		k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
									kernel0_1 << <k0_dimGrid, k0_dimBlock >> >(dev_A, dimsize, timestep, I, J, c0);
								}
								c0 += 1;
								{
									const AN5D_TYPE	__side0Len = 1;
									const AN5D_TYPE	__side1Len = 128;
									const AN5D_TYPE	__side2Len = 30;
									const AN5D_TYPE	__OlLen1 = (__halo1 * __side0Len);
									const AN5D_TYPE	__OlLen2 = (__halo2 * __side0Len);
									const AN5D_TYPE	__side1LenOl = (__side1Len + 2 * __OlLen1);
									const AN5D_TYPE	__side2LenOl = (__side2Len + 2 * __OlLen2);
									const AN5D_TYPE	__blockSize = 1 * __side2LenOl;
									assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
									dim3		k0_dimBlock(__blockSize, 1, 1);
									dim3		k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
									kernel0_1 << <k0_dimGrid, k0_dimBlock >> >(dev_A, dimsize, timestep, I, J, c0);
								}
							} else if (__c0Len % __side0LenMax == 3) {
								{
									const AN5D_TYPE	__side0Len = 2;
									const AN5D_TYPE	__side1Len = 128;
									const AN5D_TYPE	__side2Len = 28;
									const AN5D_TYPE	__OlLen1 = (__halo1 * __side0Len);
									const AN5D_TYPE	__OlLen2 = (__halo2 * __side0Len);
									const AN5D_TYPE	__side1LenOl = (__side1Len + 2 * __OlLen1);
									const AN5D_TYPE	__side2LenOl = (__side2Len + 2 * __OlLen2);
									const AN5D_TYPE	__blockSize = 1 * __side2LenOl;
									assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
									dim3		k0_dimBlock(__blockSize, 1, 1);
									dim3		k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
									kernel0_2 << <k0_dimGrid, k0_dimBlock >> >(dev_A, dimsize, timestep, I, J, c0);
								}
								c0 += 1;
								{
									const AN5D_TYPE	__side0Len = 1;
									const AN5D_TYPE	__side1Len = 128;
									const AN5D_TYPE	__side2Len = 30;
									const AN5D_TYPE	__OlLen1 = (__halo1 * __side0Len);
									const AN5D_TYPE	__OlLen2 = (__halo2 * __side0Len);
									const AN5D_TYPE	__side1LenOl = (__side1Len + 2 * __OlLen1);
									const AN5D_TYPE	__side2LenOl = (__side2Len + 2 * __OlLen2);
									const AN5D_TYPE	__blockSize = 1 * __side2LenOl;
									assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
									dim3		k0_dimBlock(__blockSize, 1, 1);
									dim3		k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
									kernel0_1 << <k0_dimGrid, k0_dimBlock >> >(dev_A, dimsize, timestep, I, J, c0);
								}
							}
						} else if (__c0Len % __side0LenMax) {
							if (__c0Len % __side0LenMax == 1) {
								const AN5D_TYPE	__side0Len = 1;
								const AN5D_TYPE	__side1Len = 128;
								const AN5D_TYPE	__side2Len = 30;
								const AN5D_TYPE	__OlLen1 = (__halo1 * __side0Len);
								const AN5D_TYPE	__OlLen2 = (__halo2 * __side0Len);
								const AN5D_TYPE	__side1LenOl = (__side1Len + 2 * __OlLen1);
								const AN5D_TYPE	__side2LenOl = (__side2Len + 2 * __OlLen2);
								const AN5D_TYPE	__blockSize = 1 * __side2LenOl;
								assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
								dim3		k0_dimBlock(__blockSize, 1, 1);
								dim3		k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
								kernel0_1 << <k0_dimGrid, k0_dimBlock >> >(dev_A, dimsize, timestep, I, J, c0);
							} else if (__c0Len % __side0LenMax == 2) {
								const AN5D_TYPE	__side0Len = 2;
								const AN5D_TYPE	__side1Len = 128;
								const AN5D_TYPE	__side2Len = 28;
								const AN5D_TYPE	__OlLen1 = (__halo1 * __side0Len);
								const AN5D_TYPE	__OlLen2 = (__halo2 * __side0Len);
								const AN5D_TYPE	__side1LenOl = (__side1Len + 2 * __OlLen1);
								const AN5D_TYPE	__side2LenOl = (__side2Len + 2 * __OlLen2);
								const AN5D_TYPE	__blockSize = 1 * __side2LenOl;
								assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
								dim3		k0_dimBlock(__blockSize, 1, 1);
								dim3		k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
								kernel0_2 << <k0_dimGrid, k0_dimBlock >> >(dev_A, dimsize, timestep, I, J, c0);
							} else if (__c0Len % __side0LenMax == 3) {
								const AN5D_TYPE	__side0Len = 3;
								const AN5D_TYPE	__side1Len = 128;
								const AN5D_TYPE	__side2Len = 26;
								const AN5D_TYPE	__OlLen1 = (__halo1 * __side0Len);
								const AN5D_TYPE	__OlLen2 = (__halo2 * __side0Len);
								const AN5D_TYPE	__side1LenOl = (__side1Len + 2 * __OlLen1);
								const AN5D_TYPE	__side2LenOl = (__side2Len + 2 * __OlLen2);
								const AN5D_TYPE	__blockSize = 1 * __side2LenOl;
								assert((__side1Len >= 2 * __side0Len * __halo1) && (__c1Len % __side1Len == 0 || __c1Len % __side1Len >= 2 * __side0Len * __halo1) && "[AN5D ERROR] Too short stream");
								dim3		k0_dimBlock(__blockSize, 1, 1);
								dim3		k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len), 1, 1);
								kernel0_3 << <k0_dimGrid, k0_dimBlock >> >(dev_A, dimsize, timestep, I, J, c0);
							}
						}
					} //END KERNEL

					cudaCheckKernel();

				} //LLL

					// shift dev_A up i rows and j columns by BENCH_RAD * BT each
          dev_A = &(dev_A_outer[(pi*PI - BENCH_RAD*BT)*dimsize + (pj*PJ - BENCH_RAD*BT)]);
			}

			{
#ifd STENCILBENCH
				cudaDeviceSynchronize();
				SB_STOP_INSTRUMENTS;
#end
				cudaCheckReturn(cudaMemcpy(A, dev_A, (size_t) (2) * (size_t) (dimsize) * (size_t) (dimsize) * sizeof(float), cudaMemcpyDeviceToHost));
			}


			for (int c0 = 0; c0 < timestep; c0 += 1)
				for (int c1 = 1; c1 < I - 1; c1 += 1) {
					j = 1;
					for (int c2 = 1; c2 < J - 1; c2 += 1)
						j = (c2 + 1);
				}
			t = 0;
			for (int c0 = 0; c0 < timestep; c0 += 1)
				t = (c0 + 1);
			for (int c0 = 0; c0 < timestep; c0 += 1) {
				i = 1;
				for (int c1 = 1; c1 < I - 1; c1 += 1)
					i = (c1 + 1);
			}
			cudaCheckReturn(cudaFree(dev_A));
		}
		//end pragma

	} else {
		for (int t = 0; t < timestep; t++)
			for (int i = BENCH_RAD; i < dimsize - BENCH_RAD; i++)
				for (int j = BENCH_RAD; j < dimsize - BENCH_RAD; j++)
					A[(t + 1) % 2][i][j] =
						A[t % 2][i - 1][j]
						+ A[t % 2][i][j - 1]
						+ A[t % 2][i][j]
						+ A[t % 2][i][j + 1]
						+ A[t % 2][i + 1][j];
	}
	return (((end_time != 0.0) ? end_time : sb_time()) - start_time);
}
