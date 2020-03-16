
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

  #pragma omp parallel for private(tj,i,j) 
  for (ti=0; ti<N; ti+=TSI)
  for (tj=0; tj<N; tj+=TSJ)
    for (i=ti; i<min(N,ti+TSI); i++)
    for (j=tj; j<min(N,tj+TSJ); j++)
        R[i*N+j] *= beta;

  #pragma omp parallel for private(tk,tj,i,k,j) 
  for (ti=0; ti<N; ti+=TSI)
  for (tk=0; tk<N; tk+=TSK)
  for (tj=0; tj<N; tj+=TSJ)
    for (i=ti; i<min(N,ti+TSI); i++)
    for (k=tk; k<min(N,tk+TSK); k++)
    for (j=tj; j<min(N,tj+TSJ); j++)
      R[i*N+j] += alpha * A[i*N+k] * B[k*N+j];

}


