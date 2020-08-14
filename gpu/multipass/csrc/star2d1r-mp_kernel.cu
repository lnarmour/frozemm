#include "star2d1r-mp_kernel.hu"
__device__ float __sbref_wrap(float *sb, size_t index) { return sb[index]; }

__global__ void kernel0_1(float *A, int dimsize, int timestep, int pi, int pj, int c0)
{
#ifndef AN5D_TYPE
#define AN5D_TYPE unsigned
#endif
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
    const AN5D_TYPE __side0Len = 1;
    const AN5D_TYPE __side1Len = 128;
    const AN5D_TYPE __side2Len = 126;
    const AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
    const AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
    const AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
    const AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
    const AN5D_TYPE __blockSize = 1 * __side2LenOl;
    const AN5D_TYPE __side1Num = (__c1Len + __side1Len - 1) / __side1Len;
    const AN5D_TYPE __side2Num = (__c2Len + __side2Len - 1) / __side2Len;
    const AN5D_TYPE __tid = threadIdx.y * blockDim.x + threadIdx.x;
    const AN5D_TYPE __local_c2 = __tid;
    const AN5D_TYPE __c1Id = blockIdx.x / __side2Num;
    const AN5D_TYPE __c2 = (blockIdx.x % __side2Num) * __side2Len + __local_c2 + __c2Pad - __OlLen2;
    float __reg_0_0;
    float __reg_0_1;
    float __reg_0_2;
    __shared__ float __b_sb_double[__blockSize * 2];
    float *__b_sb = __b_sb_double;
    const AN5D_TYPE __loadValid = 1 && __c2 >= __c2Pad - __halo2 && __c2 < __c2Pad + __c2Len + __halo2;
    const AN5D_TYPE __updateValid = 1 && __c2 >= __c2Pad && __c2 < __c2Pad + __c2Len;
    const AN5D_TYPE __writeValid1 = __updateValid && __local_c2 >= (__halo2 * 1) && __local_c2 < __side2LenOl - (__halo2 * 1);
    const AN5D_TYPE __storeValid = __writeValid1;
    AN5D_TYPE __c1;
    AN5D_TYPE __h;
    const AN5D_TYPE __c1Pad2 = __c1Pad + __side1Len * __c1Id;
    #define __LOAD(reg, h) do { if (__loadValid) { __c1 = __c1Pad2 - __halo1 + h; reg = A[((__c0 % 2) * dimsize + __c1) * dimsize + __c2]; }} while (0)
    #define __DEST (A[(((c0 + 1) % 2) * dimsize + c1) * dimsize + c2])
    #define __REGREF(reg, i2) reg
    #define __SBREF(sb, i2) __sbref_wrap(sb, (int)__tid + i2)
    #define __CALCEXPR(__rn0, __a, __b, __c) do { __rn0 = (((((__REGREF(__a, 0)) + (__SBREF(__b_sb, -1))) + (__REGREF(__b, 0))) + (__SBREF(__b_sb, 1))) + (__REGREF(__c, 0))); } while (0)
    #define __DB_SWITCH() do { __b_sb = &__b_sb_double[(__b_sb == __b_sb_double) ? __blockSize : 0]; } while (0)
    #define __CALCSETUP(a, b, c) do { __DB_SWITCH(); __b_sb[__tid] = b; __syncthreads(); } while (0)
    #define __STORE(h, reg0, reg1, reg2) do { __CALCSETUP(reg0, reg1, reg2); if (__storeValid) { __c1 = __c1Pad2 - __halo1 + h; __CALCEXPR(__DEST, reg0, reg1, reg2); } } while (0)
    if (__c1Id == 0)
    {
      __LOAD(__reg_0_0, 0);
      __LOAD(__reg_0_1, 1);
      __LOAD(__reg_0_2, 2);
      __STORE(1, __reg_0_0, __reg_0_1, __reg_0_2);
    }
    else
    {
      __LOAD(__reg_0_0, 0);
      __LOAD(__reg_0_1, 1);
      __LOAD(__reg_0_2, 2);
      __STORE(1, __reg_0_0, __reg_0_1, __reg_0_2);
    }
    __b_sb = __b_sb_double + __blockSize * 1;
    if (__c1Id == __side1Num - 1)
    {
      for (__h = 3; __h <= __c1Len - __side1Len * __c1Id + __halo1 * 2 - 3;)
      {
        __LOAD(__reg_0_0, __h);
        __STORE(__h - 1, __reg_0_1, __reg_0_2, __reg_0_0);
        __h++;
        __LOAD(__reg_0_1, __h);
        __STORE(__h - 1, __reg_0_2, __reg_0_0, __reg_0_1);
        __h++;
        __LOAD(__reg_0_2, __h);
        __STORE(__h - 1, __reg_0_0, __reg_0_1, __reg_0_2);
        __h++;
        __DB_SWITCH(); __syncthreads();
      }
      if (0) {}
      else if (__h + 0 == __c1Len - __side1Len * __c1Id + __halo1 * 2)
      {
      }
      else if (__h + 1 == __c1Len - __side1Len * __c1Id + __halo1 * 2)
      {
        __LOAD(__reg_0_0, __h + 0);
        __STORE(__h - 1, __reg_0_1, __reg_0_2, __reg_0_0);
      }
      else if (__h + 2 == __c1Len - __side1Len * __c1Id + __halo1 * 2)
      {
        __LOAD(__reg_0_0, __h + 0);
        __STORE(__h - 1, __reg_0_1, __reg_0_2, __reg_0_0);
        __LOAD(__reg_0_1, __h + 1);
        __STORE(__h + 0, __reg_0_2, __reg_0_0, __reg_0_1);
      }
    }
    else
    {
      for (__h = 3; __h <= __side1LenOl - 3;)
      {
        __LOAD(__reg_0_0, __h);
        __STORE(__h - 1, __reg_0_1, __reg_0_2, __reg_0_0);
        __h++;
        __LOAD(__reg_0_1, __h);
        __STORE(__h - 1, __reg_0_2, __reg_0_0, __reg_0_1);
        __h++;
        __LOAD(__reg_0_2, __h);
        __STORE(__h - 1, __reg_0_0, __reg_0_1, __reg_0_2);
        __h++;
        __DB_SWITCH();  __syncthreads();
      }
      if (__h == __side1LenOl) return;
      __LOAD(__reg_0_0, __h);
      __STORE(__h - 1, __reg_0_1, __reg_0_2, __reg_0_0);
      __h++;
      if (__h == __side1LenOl) return;
      __LOAD(__reg_0_1, __h);
      __STORE(__h - 1, __reg_0_2, __reg_0_0, __reg_0_1);
      __h++;
      if (__h == __side1LenOl) return;
      __LOAD(__reg_0_2, __h);
      __STORE(__h - 1, __reg_0_0, __reg_0_1, __reg_0_2);
      __h++;
    }
}
