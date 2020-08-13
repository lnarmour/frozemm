#define BENCH_DIM 1
#define BENCH_FPP 5
#define BENCH_RAD 1

#include "common.h"

#define PI 512

double kernel_stencil(SB_TYPE *A1, int compsize, int timestep, bool scop)
{
  double start_time = sb_time(), end_time = 0.0;
  int dimsize = compsize + BENCH_RAD * 2;
  SB_TYPE (*A)[dimsize] = (SB_TYPE (*)[dimsize])A1;

  int pi = 2;

  if (scop) {
#pragma scop
    for (int t = 0; t < timestep; t++)
      for (int i = pi + BENCH_RAD; i < pi + PI - BENCH_RAD; i++)
          A[(t+1)%2][i] = 0.09374f * A[t%2][i-1] + 
                          0.09376f * A[t%2][i+0] + 
                          0.09375f * A[t%2][i+1];
#pragma endscop
  }
  else {
    for (int t = 0; t < timestep; t++)
#pragma omp parallel for
      for (int i = pi + BENCH_RAD; i < pi + PI - BENCH_RAD; i++)
          A[(t+1)%2][i] = 0.09374f * A[t%2][i-1] + 
                          0.09376f * A[t%2][i+0] + 
                          0.09375f * A[t%2][i+1];
  }

  return (((end_time != 0.0) ? end_time : sb_time()) - start_time);
}
