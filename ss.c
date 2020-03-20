#include <stdio.h>
#include <omp.h>
#include "ss.h"


void MM(long N,
     PRECISION* restrict A, 
     PRECISION* restrict B, 
     PRECISION* restrict R) 
{

  long i,j,k,ti,tj,tk;
  long G;

//  #pragma omp parallel for private(ti,tj,tk,i,j,k)
  for (ti=0; ti<N; ti+=TI)
    for (tk=0; tk<N; tk+=TK)
      for (tj=0; tj<N; tj+=TJ) 
      {
        if (ti+TI<N && tj+TJ<N && tk+TK<N)
        {
          G = TI%2;
          for (i=ti; i<ti+TI-1; i+=2)
            for (k=tk; k<tk+TK; k++) {
              #pragma vector aligned
              for (j=tj; j<tj+TJ; j++) {
                R[(i+0)*N+j] += A[(i+0)*N+k] * B[k*N+j];
                R[(i+1)*N+j] += A[(i+1)*N+k] * B[k*N+j];
              }
            }
          if (G)
            for (k=tk; k<tk+TK; k++) {
              #pragma vector aligned
              for (j=tj; j<tj+TJ; j++) {
                R[(ti+TI-1)*N+j] += A[(ti+TI-1)*N+k] * B[k*N+j];
              }
            }


        } 
        else 
        {
          if (ti+TI<N)
          {
            G = TI%2;
            for (i=ti; i<ti+TI-1; i+=2)
              for (k=tk; k<min(N,tk+TK); k++) {
                #pragma vector aligned
                for (j=tj; j<min(N,tj+TJ); j++) {
                  R[(i+0)*N+j] += A[(i+0)*N+k] * B[k*N+j];
                  R[(i+1)*N+j] += A[(i+1)*N+k] * B[k*N+j];
                }
              }
            if (G)
              for (k=tk; k<min(N,tk+TK); k++) {
                #pragma vector aligned
                for (j=tj; j<min(N,tj+TJ); j++) {
                  R[(ti+TI-1)*N+j] += A[(ti+TI-1)*N+k] * B[k*N+j];
                }
              }

          }
          else
          {
            G = (N-ti)%2;
            for (i=ti; i<N-1; i+=2)
              for (k=tk; k<min(N,tk+TK); k++) {
                #pragma vector aligned
                for (j=tj; j<min(N,tj+TJ); j++) {
                  R[(i+0)*N+j] += A[(i+0)*N+k] * B[k*N+j];
                  R[(i+1)*N+j] += A[(i+1)*N+k] * B[k*N+j];
                }
              }
            if (G)
              for (k=tk; k<min(N,tk+TK); k++) {
                #pragma vector aligned
                for (j=tj; j<min(N,tj+TJ); j++) {
                  R[(N-1)*N+j] += A[(N-1)*N+k] * B[k*N+j];
                }
              }

          }

        }
      }
}

