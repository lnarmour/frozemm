#include <mkl.h>
#include <stdio.h>
#include "ss.h"


void MM_MKL(PRECISION alpha, PRECISION beta, long l, long m, long n, PRECISION* A, PRECISION* B, PRECISION* R){

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
    alpha, //multiply A by 1
    A, 
    m, //leading dimension of A
    B, 
    n, //leading dimension of B
    beta, //multiply C by this before adding AB
    R, 
    n);//leading dimension of C
	//#endif
}
