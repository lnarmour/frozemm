#include <stdio.h>
#include <stdlib.h>

#define abs(val) (val)>0.0 ? (val) : -1*(val)
#define IDX(i,j,ld) (((j)*(ld))+(i))

void printX(float *X, int M, int N) {
  for (int xj=0; xj<N; xj++) {
    for (int xi=0; xi<M; xi++) {
      printf("%3.2f ", X[IDX(xi,xj,M)]);
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
    for (i=0; i<M; i++)
      O[IDX(i,j,M)] = (float) ((i*j+1) % M) / M;

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
        
        printf("\nA\n");
        printX(A, M, K);
        printf("\nB\n");
        printX(B, K, N);
        
        printf("\nC\n");
        printX(C, M, N);
        printf("\nOracle\n");
        printX(O, M, N);

        return 1;
      }
    }
  }
  return 0;
}
