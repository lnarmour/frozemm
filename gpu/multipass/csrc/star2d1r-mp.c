#define BENCH_DIM 2
#define BENCH_FPP 9
#define BENCH_RAD 1

#define PI 896-2*BENCH_RAD
#define PJ 896-2*BENCH_RAD

#include "common.h"

double kernel_stencil(SB_TYPE *A1, int compsize, int timestep, bool scop)
{
  double start_time = sb_time(), end_time = 0.0;
  int dimsize = compsize + BENCH_RAD * 2;
  SB_TYPE (*A)[dimsize][dimsize] = (SB_TYPE (*)[dimsize][dimsize])A1;

  int pi = 2*PI;
  int pj = 2*PJ;

  if (scop) {
      #pragma scop
      for (int t = 0; t < timestep; t++)
        for (int i = pi + BENCH_RAD*(1-t); i < pi + PI + BENCH_RAD*(1-t); i++)
          for (int j = pj + BENCH_RAD*(1-t); j < pj + PJ + BENCH_RAD*(1-t); j++)
            A[(t+1)%2][i][j] =
              0.1873f * A[t%2][i-1][j]
              + 0.1876f * A[t%2][i][j-1]
              + 0.2500f * A[t%2][i][j]
              + 0.1877f * A[t%2][i][j+1]
              + 0.1874f * A[t%2][i+1][j];
      #pragma endscop
  }
  else {
    for (int t = 0; t < timestep; t++)
#pragma omp parallel for
        for (int i = pi + BENCH_RAD*(1-t); i < pi + PI + BENCH_RAD*(1-t); i++)
          for (int j = pj + BENCH_RAD*(1-t); j < pj + PJ + BENCH_RAD*(1-t); j++)
            A[(t+1)%2][i][j] =
              0.1873f * A[t%2][i-1][j]
              + 0.1876f * A[t%2][i][j-1]
              + 0.2500f * A[t%2][i][j]
              + 0.1877f * A[t%2][i][j+1]
              + 0.1874f * A[t%2][i+1][j];
  }

  return (((end_time != 0.0) ? end_time : sb_time()) - start_time);
}
