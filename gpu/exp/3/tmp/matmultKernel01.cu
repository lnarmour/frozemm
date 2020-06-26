#include "matmultKernel.h"
#include <stdio.h>

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
  float *Asub, *Bsub, *Csub;
  int num_threads=BLOCK_SIZE_X*BLOCK_SIZE_Y;
  Csub = &C.elements[C.stride * STRIP_SIZE * blockIdx.y + FOOTPRINT_SIZE_X * blockIdx.x];
  
  float Cvalues[SCALING_FACTOR] = {[0 ... SCALING_FACTOR-1]=0};
  
  // Read each strip of A and B from global DRAM into shared memory
  // perform matrix product of strips and accumate into Cvalues
  for (int m = 0;  m < (A.width / STRIP_SIZE); ++m){ 
    Asub = &A.elements[A.stride * STRIP_SIZE * m + FOOTPRINT_SIZE_Y * blockIdx.y];
    Bsub = &B.elements[B.stride * STRIP_SIZE * m + FOOTPRINT_SIZE_X * blockIdx.x];
    
    __shared__ float shared_A[FOOTPRINT_SIZE_Y][STRIP_SIZE];
    __shared__ float shared_B[STRIP_SIZE][FOOTPRINT_SIZE_X];
    
    for (int i=0; i<STRIP_SIZE; i=i+BLOCK_SIZE_Y)
      for (int j=0; j<FOOTPRINT_SIZE_X; j = j + BLOCK_SIZE_X){
        shared_A[threadIdx.x + j][threadIdx.y + i]=Asub[(threadIdx.y + i)*A.stride + (threadIdx.x + j)];
        shared_B[threadIdx.y + i][threadIdx.x + j]=Bsub[(threadIdx.y + i)*B.stride + (threadIdx.x + j)];
    }
    __syncthreads();
    
    int c=0;
    for (int i=0; i<STRIP_SIZE; i=i+(SCALING_FACTOR_Y*num_threads))
      for (int j=0; j<FOOTPRINT_SIZE_X; j=j+(SCALING_FACTOR_X*num_threads))
        for (int l=0; l<SCALING_FACTOR_X; ++l)
          for (int k=0; k<SCALING_FACTOR_Y; ++k, c++)
            for (int e=0; e <STRIP_SIZE; ++e)
              Cvalues[c] += shared_A[threadIdx.y*SCALING_FACTOR_Y + i + l][e] * shared_B[e][threadIdx.x*SCALING_FACTOR_X + j + k];

    __syncthreads();
  }


  // Write Cvalues back to global DRAM
  int d=0;
  for(int i=0; i<STRIP_SIZE; i=i+(SCALING_FACTOR_Y*num_threads))
    for (int j=0; j<FOOTPRINT_SIZE_X; j=j+(SCALING_FACTOR_X*num_threads))
      for (int l=0; l<SCALING_FACTOR_X;++l)
        for (int k=0; k<SCALING_FACTOR_Y;++k)
          Csub[(threadIdx.y*SCALING_FACTOR_Y + i + l)*C.stride+(threadIdx.x*SCALING_FACTOR_X+j)+k]=Cvalues[d++];
}
