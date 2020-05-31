#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define abs(val) (val)>0.0 ? (val) : -1*(val)
#define IDX(i,j,ld) (((j)*(ld))+(i))

extern void printX(float *X, int M, int N) {
  for (int xj=0; xj<N; xj++) {
    for (int xi=0; xi<M; xi++) {
      printf("(%d)%3.2f ", xi*N+xj, X[IDX(xi,xj,M)]);
      //printf("%3.2f ", X[IDX(xi,xj,M)]);
    }
    printf("\n");
  }
}

int check(long M, long N, long K, float *A, float *B, float *C)
{
  float *O, delta;
  long i,j,k;

  O = (float*)malloc(M * N * sizeof(float));
  for (j=0; j<N; j++) 
    for (i=0; i<M; i++) {
      O[IDX(i,j,M)] = (float) ((i*j+1) % M) / M;
    }

  #pragma omp parallel for private(k,i)
  for (j=0; j<N; j++)
    for (k=0; k<K; k++)
      for (i=0; i<M; i++)
        O[IDX(i,j,M)] += A[IDX(i,k,M)] * B[IDX(k,j,K)];

  for (j=0; j<N; j++) {
    for (i=0; i<M; i++) {
      float delta = abs(C[IDX(i,j,M)]-O[IDX(i,j,M)]);
      if (delta >= 0.001) {
        printf("Error:\n");
        printf("C(%d,%d)=%f\n", i, j, C[IDX(i,j,M)]);
        printf("Oracle(%d,%d)=%f\n", i, j, O[IDX(i,j,M)]);

        if (M<10 && N<10 && K<10) {        
          printf("\nA\n");
          printX(A, M, K);
          printf("\nB\n");
          printX(B, K, N);
          
          printf("\nC\n");
          printX(C, M, N);
          printf("\nOracle\n");
          printX(O, M, N);
        }

        return 1;
      }
    }
  }
  return 0;
}

int checkmv(long M, long K, float *A, float *B, float *C)
{
  float *O, delta;
  long i,j,k;

  O = (float*)malloc(M * sizeof(float));
  for (i=0; i<M; i++) {
    O[i] = 0.0;
  }

  for (k=0; k<K; k++)
    for (i=0; i<M; i++) {
      O[i] += A[IDX(i,k,M)] * B[k];
    }

  for (i=0; i<M; i++) {
    float delta = abs(C[i]-O[i]);
    if (delta >= 0.001) {
      printf("Error:\n");
      printf("C(%d,%d)=%f\n", i, j, C[i]);
      printf("Oracle(%d,%d)=%f\n", i, j, O[i]);

      if (M<10 && K<10) {        
        printf("\nA\n");
        printX(A, M, K);
        printf("\nB\n");
        printX(B, K, 1);
        
        printf("\nC\n");
        printX(C, M, 1);
        printf("\nOracle\n");
        printX(O, M, 1);
      }

      return 1;
    }
  }
  return 0;
}

int checkt(long M, long N, float *A, float *B, float *C)
{
  long i,j;

  for (j=0; j<N; j++) {
    for (i=0; i<M; i++) {
      float delta = abs(C[IDX(i,j,M)]-C[IDX(j,i,M)]);
      if (delta >= 0.001) {
        printf("Error:\n");
        printf("C(%d,%d)=%f\n", i, j, C[IDX(i,j,M)]);
        printf("Oracle(%d,%d)=%f\n", i, j, C[IDX(j,i,M)]);

        if (M<10 && N<10) {
          printf("\nA\n");
          printX(A, M, N);
          
          printf("\nC\n");
          printX(C, M, N);
        }

        return 1;
      }
    }
  }
  return 0;
}
