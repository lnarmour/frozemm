
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include "ss.h"

void MM(PRECISION alpha, PRECISION beta, 
     long N, long TSI, long TSJ, long TSK, 
     PRECISION* restrict A, 
     PRECISION* restrict B, 
     PRECISION* restrict R) 
{

  long i,j,k,ti,tj,tk;

  #pragma omp parallel for private(j) 
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
       R[i*N+j] *= beta;

  #pragma omp parallel for private(k,j) 
  for (i=0; i<N; i++)
    for (k=0; k<N; k++)
      for (j=0; j<N; j++)
        R[i*N+j] += alpha * A[i*N+k] * B[k*N+j];

}


