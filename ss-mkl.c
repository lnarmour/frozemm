#include <mkl.h>
#include <stdio.h>
#include "ss.h"


void MM_MKL_D(long l, long m, long n, PRECISION* A, PRECISION* B, PRECISION* R, long ti, long tj, long tk ){

	for (int i=0; i<l; i++)
		for (int j=0; j<m; j++)
			if (A[i*m+j] == 0) {
				printf("AHH --> (%ld,%ld,%ld)\n", ti, tj, tk);
			}

	MM_MKL(l,m,n,A,B,R);
}


void MM_MKL(long l, long m, long n, PRECISION* A, PRECISION* B, PRECISION* R){

	//for (int i=0; i<l; i++)
	//for (int k=0; k<m; k++)
	//for (int j=0; j<n; j++) {
	//	R[i*n+j] += A[i*m+k] * B[k*n+j];
	//	//printf("%f, %f --> %f\n", A[i*m+k], B[k*n+j], R[i*n+j]);
	//}
	//#ifdef INT
	//#else
  CBLAS_GEMM(CblasRowMajor, //storage format: row major
    CblasNoTrans, //use A, not A transpose
    CblasNoTrans, //Use B, not B transpose
    l, //rows in A
    n, //columns in B
    m, //columns in A/rows in B
    1, //multiply A by 1
    A, 
    m, //leading dimension of A
    B, 
    n, //leading dimension of B
    1, //multiply C by this before adding AB
    R, 
    n);//leading dimension of C
	//#endif
}
