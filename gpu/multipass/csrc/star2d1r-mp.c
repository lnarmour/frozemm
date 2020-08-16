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

  
  int t, i, j;
  int I = PI; 
  int J = PJ; 

  if (scop) {
    #pragma scop
    for (t = 0; t < timestep; t++) 
      for (i = BENCH_RAD; i < I - BENCH_RAD; i++)
        for (j = BENCH_RAD; j < J - BENCH_RAD; j++)
          A[(t+1)%2][i][j] =
            A[t%2][i-1][j]
            + A[t%2][i][j-1]
            + A[t%2][i][j]
            + A[t%2][i][j+1]
            + A[t%2][i+1][j];
    #pragma endscop
    // end pragma

  }
  else {
    for (int t = 0; t < timestep; t++) 
      for (int i = BENCH_RAD; i < dimsize-BENCH_RAD; i++)
        for (int j = BENCH_RAD; j < dimsize-BENCH_RAD; j++)
          A[(t+1)%2][i][j] =
            A[t%2][i-1][j]
            + A[t%2][i][j-1]
            + A[t%2][i][j]
            + A[t%2][i][j+1]
            + A[t%2][i+1][j];
  }
  return (((end_time != 0.0) ? end_time : sb_time()) - start_time);
}
