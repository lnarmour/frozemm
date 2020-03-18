#include "ss.h"

void MM(PRECISION alpha, PRECISION beta, 
     long N, long TSI, long TSJ, long TSK, 
     PRECISION* restrict A, 
     PRECISION* restrict B, 
     PRECISION* restrict R) 
{

  long i,j,k,ti,tj,tk;

  for (i=0; i<N; i++)
    for (k=0; k<N; k++) {
      #pragma vector aligned
      for (j=0; j<N; j++)
        R[i*N+j] += A[i*N+k] * B[k*N+j];
    }
}


