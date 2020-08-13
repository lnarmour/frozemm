#include "star2d1r-128-4-128_kernel.hu"
__global__ void kernel0(float *A, int dimsize, int timestep)
{

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    for (int c0 = ppcg_max(0, -dimsize + 1026); c0 < timestep; c0 += 1)
      for (int c1 = 1025; c1 <= ppcg_min(1534, dimsize + c0 - 1); c1 += 1)
        for (int c2 = 1025; c2 <= ppcg_min(1534, dimsize + c0 - 1); c2 += 1)
          A[(((c0 + 1) % 2) * dimsize + (-c0 + c1)) * dimsize + (-c0 + c2)] = (((((0.1873f * A[((c0 % 2) * dimsize + (-c0 + c1 - 1)) * dimsize + (-c0 + c2)]) + (0.1876f * A[((c0 % 2) * dimsize + (-c0 + c1)) * dimsize + (-c0 + c2 - 1)])) + (0.2500f * A[((c0 % 2) * dimsize + (-c0 + c1)) * dimsize + (-c0 + c2)])) + (0.1877f * A[((c0 % 2) * dimsize + (-c0 + c1)) * dimsize + (-c0 + c2 + 1)])) + (0.1874f * A[((c0 % 2) * dimsize + (-c0 + c1 + 1)) * dimsize + (-c0 + c2)]));
}
