#include <stdio.h>
#include <stdlib.h>

#define abs(val) (val)>0.0 ? (val) : -1*(val)

void* xmalloc(size_t);

void printM(float *M, int N) {
  for (int xi=0; xi<N; xi++) {
    for (int xj=0; xj<N; xj++) {
      printf("%3.2f ", M[xi*N+xj]);
    }
    printf("\n");
  }
}


int check(long N, float *A, float *B, float *C)
{
  float *O, delta;
  long i,j,k;

  O = xmalloc(N * N * sizeof(float));
  for (i=0; i<N; i++)
    for (j=0; j<N; j++) 
      O[i*N+j] = 0;

  for (i=0; i<N; i++)
    for (k=0; k<N; k++)
      for (j=0; j<N; j++)
        O[i*N+j] += A[i*N+k] * B[k*N+j];

  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      float delta = abs(C[i*N+j]-O[i*N+j]);
      if (delta >= 0.001) {
        printf("Error:\n");
        printf("C(%d,%d)=%f\n", i, j, C[i*N+j]);
        printf("Oracle(%d,%d)=%f\n", i, j, O[i*N+j]);
        
        printf("\nA\n");
        printM(A, N);
        printf("\nB\n");
        printM(B, N);
        
        printf("\nC\n");
        printM(C, N);
        printf("\nOracle\n");
        printM(O, N);

        return 1;
      }
    }
  }
  return 0;
}
