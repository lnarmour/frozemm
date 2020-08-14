#define BENCH_DIM 2
#define BENCH_FPP 9
#define BENCH_RAD 1

#define PI 896
#define PJ 896

#include "common.h"

double kernel_stencil(SB_TYPE *A1, int compsize, int timestep, bool scop)
{
  double start_time = sb_time(), end_time = 0.0;
  int dimsize = compsize + BENCH_RAD * 2;
  SB_TYPE (*A)[dimsize][dimsize] = (SB_TYPE (*)[dimsize][dimsize])A1;

  if (scop) {
    for (int pi=BENCH_RAD; pi<dimsize + (timestep-1)*BENCH_RAD; pi+=PI)
      for (int pj=BENCH_RAD; pj<dimsize + (timestep-1)*BENCH_RAD; pj+=PJ) {
        #pragma scop
        for (int t = 0; t < timestep; t++) 
          for (int i = max(BENCH_RAD, pi + BENCH_RAD*(-t)); i < min(pi + PI + BENCH_RAD*(-t), dimsize-BENCH_RAD); i++)
            for (int j = max(BENCH_RAD, pj + BENCH_RAD*(-t)); j < min(pj + PJ + BENCH_RAD*(-t), dimsize-BENCH_RAD); j++)
              A[(t+1)%2][i][j] =
                A[t%2][i-1][j]
                + A[t%2][i][j-1]
                + A[t%2][i][j]
                + A[t%2][i][j+1]
                + A[t%2][i+1][j];
        #pragma endscop
      }
  }
  else {
    for (int pi=BENCH_RAD; pi<dimsize + (timestep-1)*BENCH_RAD; pi+=PI)
      for (int pj=BENCH_RAD; pj<dimsize + (timestep-1)*BENCH_RAD; pj+=PJ)
        for (int t = 0; t < timestep; t++) 
          for (int i = max(BENCH_RAD, pi + BENCH_RAD*(-t)); i < min(pi + PI + BENCH_RAD*(-t), dimsize-BENCH_RAD); i++)
            for (int j = max(BENCH_RAD, pj + BENCH_RAD*(-t)); j < min(pj + PJ + BENCH_RAD*(-t), dimsize-BENCH_RAD); j++)
              A[(t+1)%2][i][j] =
                A[t%2][i-1][j]
                + A[t%2][i][j-1]
                + A[t%2][i][j]
                + A[t%2][i][j+1]
                + A[t%2][i+1][j];
  }
  return (((end_time != 0.0) ? end_time : sb_time()) - start_time);
}
