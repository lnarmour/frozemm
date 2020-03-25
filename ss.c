#include <stdio.h>
#include "ss.h"

#define N 2000

void MM(long M,
     PRECISION* restrict x0, 
     PRECISION* restrict y0, 
     PRECISION* restrict z0) 
{

  long i,j,k;

  // fma.MM2.c version30   B_transpose (y0)
  // shared_writes(False) shared_reads(False) unroll(1,2,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
      }
    }
  }



}
