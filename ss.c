#include <stdio.h>
#include "ss.h"


void MM(long N,
     PRECISION* restrict A, 
     PRECISION* restrict B, 
     PRECISION* restrict R) 
{

  long i,j,k,ti,tj,tk;

  // case 1 of 8: ti,tj,tk = full,full,full
  for (ti=0; ti<N-TI; ti+=TI)
    for (tk=0; tk<N-TK; tk+=TK)
      for (tj=0; tj<N-TJ; tj+=TJ)
        for (i=ti; i<ti+TI; i++)
          for (k=tk; k<tk+TK; k++) {
            #pragma vector aligned
            for (j=tj; j<tj+TJ; j++) {
              R[i*N+j] += A[i*N+k] * B[k*N+j];
            }
          }

  // case 2 of 8: ti,tj,tk = full,full,partial
  for (ti=0; ti<N-TI; ti+=TI)
    for (tj=0; tj<N-TJ; tj+=TJ)
      for (i=ti; i<ti+TI; i++)
        for (k=N-N%TK; k<N; k++) {
          #pragma vector aligned
          for (j=tj; j<tj+TJ; j++) {
            R[i*N+j] += A[i*N+k] * B[k*N+j];
          }
        }

  // case 3 of 8: ti,tj,tk = full,partial,full
  for (ti=0; ti<N-TI; ti+=TI)
    for (tk=0; tk<N-TK; tk+=TK)
      for (i=ti; i<ti+TI; i++)
        for (k=tk; k<tk+TK; k++) {
          #pragma vector aligned
          for (j=N-N%TJ; j<N; j++) {
            R[i*N+j] += A[i*N+k] * B[k*N+j];
          }
        }

  // case 4 of 8: ti,tj,tk = full,partial,partial
  for (ti=0; ti<N-TI; ti+=TI)
    for (i=ti; i<ti+TI; i++)
      for (k=N-N%TK; k<N; k++) {
        #pragma vector aligned
        for (j=N-N%TJ; j<N; j++) {
          R[i*N+j] += A[i*N+k] * B[k*N+j];
        }
      }

  // case 5 of 8: ti,tj,tk = partial,full,full
  for (tk=0; tk<N-TK; tk+=TK)
    for (tj=0; tj<N-TJ; tj+=TJ)
      for (i=N-N%TI; i<N; i++)
        for (k=tk; k<tk+TK; k++) {
          #pragma vector aligned
          for (j=tj; j<tj+TJ; j++) {
            R[i*N+j] += A[i*N+k] * B[k*N+j];
          }
        }

  // case 6 of 8: ti,tj,tk = partial,full,partial
  for (tj=0; tj<N-TJ; tj+=TJ)
    for (i=N-N%TI; i<N; i++)
      for (k=N-N%TK; k<N; k++) {
        #pragma vector aligned
        for (j=tj; j<tj+TJ; j++) {
          R[i*N+j] += A[i*N+k] * B[k*N+j];
        }
      }

  // case 7 of 8: ti,tj,tk = partial,partial,full
  for (tk=0; tk<N-TK; tk+=TK)
    for (i=N-N%TI; i<N; i++)
      for (k=tk; k<tk+TK; k++) {
        #pragma vector aligned
        for (j=N-N%TJ; j<N; j++) {
          R[i*N+j] += A[i*N+k] * B[k*N+j];
        }
      }

  // case 8 of 8: ti,tj,tk = partial,partial,partial
  for (i=N-N%TI; i<N; i++)
    for (k=N-N%TK; k<N; k++) {
      #pragma vector aligned
      for (j=N-N%TJ; j<N; j++) {
        R[i*N+j] += A[i*N+k] * B[k*N+j];
      }
    }


}

