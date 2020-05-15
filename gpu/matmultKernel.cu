///
/// each thread loads P values from A and P values from B. Let P = 4 for below example.
/// map thread (i,j) to a value k, where 0<=k<256 (BLOCK_SIZE*BLOCK_SIZE)
/// 
/// k = i * BLOCK_SIZE + j 
///
/// the first thread (0,0), k=0, in a block loads these elements:
/// - Asub(0,0)   Bsub(0,0)
/// - Asub(8,0)   Bsub(8,0) 
/// - Asub(16,0)  Bsub(16,0) 
/// - Asub(24,0)  Bsub(24,0) 
///
/// the second thread (0,1), k=1, loads these:
/// - Asub(0,1)   Bsub(0,1)
/// - Asub(8,1)   Bsub(8,1) 
/// - Asub(16,1)  Bsub(16,1) 
/// - Asub(24,1)  Bsub(24,1) 
///
/// the 91st thread (5,10), k=90, loads these:
/// - Asub(2,26)   Bsub(2,26)
/// - Asub(10,26)  Bsub(10,26) 
/// - Asub(18,26)  Bsub(18,26) 
/// - Asub(26,26)  Bsub(26,26) 
///
/// the k'th thread (k/32, k%32), loads these:
/// - Asub( 0+k/32, k%32)   Bsub( 0+k/32, k%32)
/// - Asub( 8+k/32, k%32)   Bsub( 8+k/32, k%32)
/// - Asub(16+k/32, k%32)   Bsub(16+k/32, k%32)
/// - Asub(24+k/32, k%32)   Bsub(24+k/32, k%32)
///


#include "matmultKernel.h"

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

  float *Asub, *Bsub, *Csub;
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];

	int k = thread_row * BLOCK_SIZE + thread_col;
	int i = k / FOOTPRINT_SIZE;
	int j = k % FOOTPRINT_SIZE;

  float Cvalue[P];
#pragma unroll
	for (int p=0; p<P; p++) {
		Cvalue[p] = 0;
	}

  for (int m = 0;  m < (A.width / FOOTPRINT_SIZE); ++m){
    Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * m];
    Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_col];

    __shared__ float shared_A[P][FOOTPRINT_SIZE/P][FOOTPRINT_SIZE];
    __shared__ float shared_B[P][FOOTPRINT_SIZE/P][FOOTPRINT_SIZE];

#pragma unroll
		for (int p=0; p<P; p++) {
			shared_A[p][i][j] = Asub[((FOOTPRINT_SIZE/P) * p + i) * A.stride + j]; 
			shared_B[p][i][j] = Bsub[((FOOTPRINT_SIZE/P) * p + i) * B.stride + j];
		}

    __syncthreads();

#pragma unroll
		for (int p=0; p<P; p++)
#pragma unroll
	   	for(int e=0; e<FOOTPRINT_SIZE; ++e)
				Cvalue[p] += shared_A[p][i][e] * shared_B[e/(FOOTPRINT_SIZE/P)][e%(FOOTPRINT_SIZE/P)][j];

    __syncthreads();
  }

#pragma unroll
	for (int p=0; p<P; p++)
		Csub[((FOOTPRINT_SIZE/P) * p + i) * C.stride + j] = Cvalue[p];
}

