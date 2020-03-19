#include <stdio.h>
#include "ss.h"


void MM(long N,
     PRECISION* restrict A, 
     PRECISION* restrict B, 
     PRECISION* restrict R) 
{

  long i,j,k,ti,tj,tk;

  // case 1 of 8: ti,tj,tk = full,full,full
  for (ti=0; ti<N; ti+=TI)
    for (tk=0; tk<N; tk+=TK)
      for (tj=0; tj<N; tj+=TJ) {

        if (ti+TI<N && tj+TJ<N && tk+TK<N) {
          for (i=ti; i<ti+TI; i++)
            for (k=tk; k<tk+TK; k++) {
              #pragma vector aligned
              for (j=tj; j<tj+TJ; j++) {
                R[i*N+j] += A[i*N+k] * B[k*N+j];
              }
            }
        } else {
          for (i=ti; i<min(N,ti+TI); i++)
            for (k=tk; k<min(N,tk+TK); k++) {
              #pragma vector aligned
              for (j=tj; j<min(N,tj+TJ); j++) {
                R[i*N+j] += A[i*N+k] * B[k*N+j];
              }
            }
        }

      }
}

