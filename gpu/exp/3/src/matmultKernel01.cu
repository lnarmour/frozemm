#include "matmultKernel.h"
#include <stdio.h>

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
  const int num_threads = blockDim.x * blockDim.y;
  const int num_Cvalues = SCALING_FACTOR_X * SCALING_FACTOR_Y; 

  float *Asub, *Bsub, *Csub;
  float Cvalues[num_Cvalues];

  Csub = &C.elements[C.stride * FOOTPRINT_SIZE_Y * blockIdx.y + FOOTPRINT_SIZE_X * blockIdx.x];
  for (int c=0; c<num_Cvalues; c++)
    Cvalues[c] = 0;
  
  // Read each strip of A and B from global DRAM into shared memory
  // perform matrix product of strips and accumate into Cvalues
  for (int m = 0;  m < (A.width / STRIP_SIZE); ++m){ 
    Asub = &A.elements[A.stride * STRIP_SIZE * m + FOOTPRINT_SIZE_Y * blockIdx.y];
    Bsub = &B.elements[B.stride * STRIP_SIZE * m + FOOTPRINT_SIZE_X * blockIdx.x];
    
    __shared__ float shared_A[STRIP_SIZE][FOOTPRINT_SIZE_Y];
    __shared__ float shared_B[STRIP_SIZE][FOOTPRINT_SIZE_X];
   
    // transpose the TT strip of Asub into SS strip in shared_A
    for (int i=threadIdx.y; i<STRIP_SIZE; i+=BLOCK_SIZE_Y)
      for (int j=threadIdx.x; j<FOOTPRINT_SIZE_X; j+=BLOCK_SIZE_X){
        shared_A[i][j] = Asub[i*A.stride + j];
        shared_B[i][j] = Bsub[i*B.stride + j];
    }
    __syncthreads();
    
    int c=0;
    for (int i=0; i<STRIP_SIZE; i=i+(SCALING_FACTOR_Y*num_threads))
      for (int j=0; j<FOOTPRINT_SIZE_X; j=j+(SCALING_FACTOR_X*num_threads))
        for (int l=0; l<SCALING_FACTOR_X; ++l)
          for (int k=0; k<SCALING_FACTOR_Y; ++k, c++) {
            for (int e=0; e <STRIP_SIZE; ++e) {
              float _a = shared_A[e][threadIdx.y*SCALING_FACTOR_Y + i + l];
              float _b = shared_B[e][threadIdx.x*SCALING_FACTOR_X + j + k];
              Cvalues[c] += _a * _b;
            }
          }
    __syncthreads();
  }

  // Write Cvalues back to global DRAM
  int d=0;
  for(int i=0; i<STRIP_SIZE; i=i+(SCALING_FACTOR_Y*num_threads))
    for (int j=0; j<FOOTPRINT_SIZE_X; j=j+(SCALING_FACTOR_X*num_threads))
      for (int l=0; l<SCALING_FACTOR_X;++l)
        for (int k=0; k<SCALING_FACTOR_Y;++k) {
          int idx = (threadIdx.y*SCALING_FACTOR_Y+i+l)*C.stride + (threadIdx.x*SCALING_FACTOR_X+j) + k;
          Csub[idx] = Cvalues[d++];
        }
}
