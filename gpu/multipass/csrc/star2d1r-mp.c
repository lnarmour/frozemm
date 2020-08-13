#define BENCH_DIM 2
#define BENCH_FPP 5
#define BENCH_RAD 1

#define PI 128
#define PJ 128

#include "common.h"

double kernel_stencil(SB_TYPE *A1, int compsize, int timestep, bool scop)
{
  double start_time = sb_time(), end_time = 0.0;
  int dimsize = compsize + BENCH_RAD * 2;
  SB_TYPE (*A)[dimsize][dimsize] = (SB_TYPE (*)[dimsize][dimsize])A1;

  int pi = 2;
  int pj = 2;

  if (scop) {

  for (int pi = BENCH_RAD; pi < dimsize - BENCH_RAD; pi += PI)
  for (int pj = BENCH_RAD; pj < dimsize - BENCH_RAD; pj += PJ)
  {
#pragma scop
    for (int t = 0; t < timestep; t++)
      for (int i = pi - BENCH_RAD * t; i < pi + PI - BENCH_RAD * t; i++)
        for (int j = pj - BENCH_RAD * t; j < pj + PJ - BENCH_RAD * t; j++)
          A[(t+1)%2][i][j] =
            0.1873f * A[t%2][i-1][j]
            + 0.1876f * A[t%2][i][j-1]
            + 0.2500f * A[t%2][i][j]
            + 0.1877f * A[t%2][i][j+1]
            + 0.1874f * A[t%2][i+1][j];
#pragma endscop
  }


  }
  else {
    for (int t = 0; t < timestep; t++)
#pragma omp parallel for
      for (int i = BENCH_RAD; i < dimsize - BENCH_RAD; i++)
        for (int j = BENCH_RAD; j < dimsize - BENCH_RAD; j++)
          A[(t+1)%2][i][j] =
            0.1873f * A[t%2][i-1][j]
            + 0.1876f * A[t%2][i][j-1]
            + 0.2500f * A[t%2][i][j]
            + 0.1877f * A[t%2][i][j+1]
            + 0.1874f * A[t%2][i+1][j];
  }

  return (((end_time != 0.0) ? end_time : sb_time()) - start_time);
}
