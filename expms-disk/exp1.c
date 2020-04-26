#include "exp.h"

void kernel(long N, double *X)
{
  long i;
  for (i=0; i<N*N; i++)
    X[i] = 0.0;
}

int main(int argc, char** argv)
{
  // intended to be run on 16 GB machine
  // 65K by 65K doubles takes ~33 GBs
  long N = 65000;
  double *X = xmalloc(N * N * sizeof(double));

  start_timer();
  kernel(N,X);
  stop_timer();

  printf("%f\n", elapsed_time);
}
