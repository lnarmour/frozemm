// perf stat -e page-faults:u,major-faults:u,minor-faults:u ./MM 10000 5000 5000 5000

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include "ss.h"


void MM(PRECISION alpha, PRECISION beta, 
     long N, long TSI, long TSJ, long TSK, 
     PRECISION* restrict A, 
     PRECISION* restrict B, 
     PRECISION* restrict R, 
     double times[3])
{

	struct timeval time;
	long i,j,k,ti,tj,tk;

	start_timer(1);

  #pragma omp parallel for private(j) 
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      R[i*N+j] *= beta;

  #pragma omp parallel for private(k,j)
  for (i=0; i<N; i++)
    for (k=0; k<N; k++)
      for (j=0; j<N; j++)
        R[i*N+j] += alpha * A[i*N+k] * B[k*N+j];

	stop_timer(1);

}


