#include "star3d2r-32x32-2-128_kernel.hu"
__global__ void kernel0(float *A, int dimsize, int timestep)
{

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    for (int c0 = ppcg_max(0, -dimsize + dimsize / 2 + 146); c0 < timestep; c0 += 1)
      for (int c1 = 290; c1 <= ppcg_min(381, dimsize + 2 * c0 - 1); c1 += 1)
        for (int c2 = 290; c2 <= ppcg_min(381, dimsize + 2 * c0 - 1); c2 += 1)
          for (int c3 = 290; c3 <= ppcg_min(381, dimsize + 2 * c0 - 1); c3 += 1)
            A[((((c0 + 1) % 2) * dimsize + (-2 * c0 + c1)) * dimsize + (-2 * c0 + c2)) * dimsize + (-2 * c0 + c3)] = (((((((((((((0.2500f * A[(((c0 % 2) * dimsize + (-2 * c0 + c1)) * dimsize + (-2 * c0 + c2)) * dimsize + (-2 * c0 + c3)]) + (0.0620f * A[(((c0 % 2) * dimsize + (c1 - 1)) * dimsize + c2) * dimsize + c3])) + (0.0621f * A[(((c0 % 2) * dimsize + (-2 * c0 + c1 + 1)) * dimsize + (-2 * c0 + c2)) * dimsize + (-2 * c0 + c3)])) + (0.0622f * A[(((c0 % 2) * dimsize + c1) * dimsize + (c2 - 1)) * dimsize + c3])) + (0.0623f * A[(((c0 % 2) * dimsize + (-2 * c0 + c1)) * dimsize + (-2 * c0 + c2 + 1)) * dimsize + (-2 * c0 + c3)])) + (0.0624f * A[(((c0 % 2) * dimsize + c1) * dimsize + c2) * dimsize + (c3 - 1)])) + (0.06245f * A[(((c0 % 2) * dimsize + (-2 * c0 + c1)) * dimsize + (-2 * c0 + c2)) * dimsize + (-2 * c0 + c3 + 1)])) + (0.06255f * A[(((c0 % 2) * dimsize + (c1 - 2)) * dimsize + c2) * dimsize + c3])) + (0.0626f * A[(((c0 % 2) * dimsize + (-2 * c0 + c1 + 2)) * dimsize + (-2 * c0 + c2)) * dimsize + (-2 * c0 + c3)])) + (0.0627f * A[(((c0 % 2) * dimsize + c1) * dimsize + (c2 - 2)) * dimsize + c3])) + (0.0628f * A[(((c0 % 2) * dimsize + (-2 * c0 + c1)) * dimsize + (-2 * c0 + c2 + 2)) * dimsize + (-2 * c0 + c3)])) + (0.0629f * A[(((c0 % 2) * dimsize + c1) * dimsize + c2) * dimsize + (c3 - 2)])) + (0.0630f * A[(((c0 % 2) * dimsize + (-2 * c0 + c1)) * dimsize + (-2 * c0 + c2)) * dimsize + (-2 * c0 + c3 + 2)]));
}
