#define BENCH_DIM 2
#define BENCH_FPP 9
#define BENCH_RAD 1

#include "common.h"

#define PI 512
#define PJ 512
#define pi 2
#define pj 2

double kernel_stencil(SB_TYPE *A1, int compsize, int timestep, bool scop)
{
  double start_time = sb_time(), end_time = 0.0;
  int dimsize = compsize + BENCH_RAD * 2;
  SB_TYPE (*A)[dimsize][dimsize] = (SB_TYPE (*)[dimsize][dimsize])A1;

  if (scop) {
#pragma scop
    for (int t = 0; t < timestep; t++)
      for (int i = pi*PI+BENCH_RAD; i < (pi+1)*PI-BENCH_RAD; i++)
        for (int j = pj*PJ+BENCH_RAD; j < (pj+1)*PJ-BENCH_RAD; j++)
          A[(t+1)%2][i-BENCH_RAD*t][j-BENCH_RAD*t] =
            0.1873f * A[t%2][i-1-BENCH_RAD*t][j-BENCH_RAD*t]
            + 0.1876f * A[t%2][i-BENCH_RAD*t][j-1-BENCH_RAD*t]
            + 0.2500f * A[t%2][i-BENCH_RAD*t][j-BENCH_RAD*t]
            + 0.1877f * A[t%2][i-BENCH_RAD*t][j+1-BENCH_RAD*t]
            + 0.1874f * A[t%2][i+1-BENCH_RAD*t][j-BENCH_RAD*t];
#pragma endscop
  }
  else {
    for (int t = 0; t < timestep; t++)
#pragma omp parallel for
      for (int i = pi*PI+BENCH_RAD; i < (pi+1)*PI-BENCH_RAD; i++)
        for (int j = pj*PJ+BENCH_RAD; j < (pj+1)*PJ-BENCH_RAD; j++)
          A[(t+1)%2][i-BENCH_RAD*t][j-BENCH_RAD*t] =
            0.1873f * A[t%2][i-1-BENCH_RAD*t][j-BENCH_RAD*t]
            + 0.1876f * A[t%2][i-BENCH_RAD*t][j-1-BENCH_RAD*t]
            + 0.2500f * A[t%2][i-BENCH_RAD*t][j-BENCH_RAD*t]
            + 0.1877f * A[t%2][i-BENCH_RAD*t][j+1-BENCH_RAD*t]
            + 0.1874f * A[t%2][i+1-BENCH_RAD*t][j-BENCH_RAD*t];
  }

  return (((end_time != 0.0) ? end_time : sb_time()) - start_time);
}
