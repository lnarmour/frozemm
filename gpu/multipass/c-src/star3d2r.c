#define BENCH_DIM 3
#define BENCH_FPP 25
#define BENCH_RAD 2

#include "common.h"

#define PI 100
#define PJ 100
#define PK 100

double kernel_stencil(SB_TYPE *A1, int compsize, int timestep, bool scop)
{
  double start_time = sb_time(), end_time = 0.0;
  int dimsize = compsize + BENCH_RAD * 2;
  SB_TYPE (*A)[dimsize][dimsize][dimsize]
    = (SB_TYPE (*)[dimsize][dimsize][dimsize])A1;

  int pi = 2;
  int pj = 2;
  int pk = 2;

  if (scop) {
#pragma scop
    for (int t = 0; t < timestep; t++)
      for (int i = pi*PI; i < (pi+1)*PI; i++)
        for (int j = pj*PJ; j < (pj+1)*PJ; j++)
          for (int k = pk*PK; k < (pk+1)*PK; k++)
            A[(t+1)%2][i][j][k] =
              0.2500f * A[t%2][i][j][k]
              + 0.0620f * A[t%2][i-1][j][k] + 0.0621f * A[t%2][i+1][j][k]
              + 0.0622f * A[t%2][i][j-1][k] + 0.0623f * A[t%2][i][j+1][k]
              + 0.0624f * A[t%2][i][j][k-1] + 0.06245f * A[t%2][i][j][k+1]

              + 0.06255f * A[t%2][i-2][j][k] + 0.0626f * A[t%2][i+2][j][k]
              + 0.0627f  * A[t%2][i][j-2][k] + 0.0628f * A[t%2][i][j+2][k]
              + 0.0629f  * A[t%2][i][j][k-2] + 0.0630f * A[t%2][i][j][k+2];
#pragma endscop
  }
  else {
    for (int t = 0; t < timestep; t++)
#pragma omp parallel for
      for (int i = pi*PI; i < (pi+1)*PI; i++)
        for (int j = pj*PJ; j < (pj+1)*PJ; j++)
          for (int k = pk*PK; k < (pk+1)*PK; k++)
            A[(t+1)%2][i][j][k] =
              0.2500f * A[t%2][i][j][k]
              + 0.0620f * A[t%2][i-1][j][k] + 0.0621f * A[t%2][i+1][j][k]
              + 0.0622f * A[t%2][i][j-1][k] + 0.0623f * A[t%2][i][j+1][k]
              + 0.0624f * A[t%2][i][j][k-1] + 0.06245f * A[t%2][i][j][k+1]

              + 0.06255f * A[t%2][i-2][j][k] + 0.0626f * A[t%2][i+2][j][k]
              + 0.0627f  * A[t%2][i][j-2][k] + 0.0628f * A[t%2][i][j+2][k]
              + 0.0629f  * A[t%2][i][j][k-2] + 0.0630f * A[t%2][i][j][k+2];
  }

  return (((end_time != 0.0) ? end_time : sb_time()) - start_time);
}
