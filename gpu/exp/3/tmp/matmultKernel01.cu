///
/// matmultKernel00.cu
/// For CSU CS575 Spring 2011
/// Instructor: Wim Bohm
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-23 DVN
///
/// Multiplies two matrices using CUDA: A x B = C
///
/// Copy this file and modify the MatMultKernel device function for
/// each of your experiments. 
///

#include "matmultKernel.h"

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
  // matrix blocks
  float *Asub, *Bsub, *Csub;
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;
  int num_threads=BLOCK_SIZE_X*BLOCK_SIZE_Y;

  Csub = &C.elements[C.stride * STRIP_SIZE * block_row + FOOTPRINT_SIZE_X * block_col];

  float Cvalues[SCALING_FACTOR] = {[0 ... SCALING_FACTOR-1]=0};


 
  for (int m = 0;  m < (A.width / STRIP_SIZE); ++m) {
    Asub = &A.elements[A.stride * STRIP_SIZE * m + FOOTPRINT_SIZE_Y * block_row];
    Bsub = &B.elements[B.stride * STRIP_SIZE * m + FOOTPRINT_SIZE_X * block_col];
    __shared__ float shared_A[FOOTPRINT_SIZE_Y][STRIP_SIZE];
    __shared__ float shared_B[STRIP_SIZE][FOOTPRINT_SIZE_X];
    
    for (int i=0; i<STRIP_SIZE; i=i+BLOCK_SIZE_Y) {
      for (int j=0; j<FOOTPRINT_SIZE_X; j = j + BLOCK_SIZE_X) {
        shared_A[thread_col + j][thread_row + i]=Asub[(thread_row + i)*A.stride + (thread_col + j)];
        shared_B[thread_row + i][thread_col + j]=Bsub[(thread_row + i)*B.stride + (thread_col + j)];
      }
    }

    __syncthreads();
    
    // Do an inproduct of one row of shared_A and one col of shared_B
    // computing one Cvalue by accumulation
    if (MODE == 0) {
      int c=0;
      #pragma unroll
      for (int i=0; i<STRIP_SIZE; i=i+(BLOCK_SIZE_Y*num_threads)) {
        for (int j=0; j<FOOTPRINT_SIZE_X; j=j+(BLOCK_SIZE_X*num_threads)) {
          for (int l=0; l<BLOCK_SIZE_Y; ++l) {
            for (int k=0; k<BLOCK_SIZE_X; ++k) {
              for (int e=0; e<STRIP_SIZE; ++e) {
                Cvalues[c] += shared_A[(thread_row * BLOCK_SIZE_Y)+i+l][e] * shared_B[e][(thread_col * BLOCK_SIZE_X)+j+k];
              }
              c=c+1;
            }
          }
        }
      }
      __syncthreads();
    }
    if (MODE == 1) {
      int c=0;
      #pragma unroll
      for (int i=0; i<STRIP_SIZE; i=i+(BLOCK_SIZE_X)) {
        for (int j=0; j<FOOTPRINT_SIZE_X; j=j+(BLOCK_SIZE_X)) {
          for (int e=0; e<STRIP_SIZE; ++e) {
            Cvalues[c] += shared_A[thread_row + i][e] * shared_B[e][thread_col + j];
          }
          c=c+1;
        }
      }
      __syncthreads();
    }
    if (MODE == 2) {
      int c=0;
      #pragma unroll
      for (int i=0; i<STRIP_SIZE; i=i+(SCALING_FACTOR_Y*num_threads)) {
        for (int j=0; j<FOOTPRINT_SIZE_X; j=j+(SCALING_FACTOR_X*num_threads)) {
          for (int l=0; l<SCALING_FACTOR_X; ++l) {
            for (int k=0; k<SCALING_FACTOR_Y; ++k) {
              for (int e=0; e <STRIP_SIZE; ++e) {
                Cvalues[c] += shared_A[thread_row*SCALING_FACTOR_Y + i + l][e] * shared_B[e][thread_col*SCALING_FACTOR_X + j + k];
              }
              c=c+1;
            }
          }
        }
      }
      __syncthreads();
    }
  }


  int d=0;
  if (MODE == 0) {
    for (int i=0; i<STRIP_SIZE; i=i+(BLOCK_SIZE_Y*num_threads)) {
      for (int j=0; j<FOOTPRINT_SIZE_X; j=j+(BLOCK_SIZE_X*num_threads)) {
        for (int l=0; l<BLOCK_SIZE_Y; ++l) {
          for (int k=0; k<BLOCK_SIZE_X; ++k) {
            Csub[((thread_row*BLOCK_SIZE_Y)+i+l)*C.stride + (thread_col*BLOCK_SIZE_X)+j+k] = Cvalues[d];
            d=d+1;
          }
        }
      }
    }
  }

  if (MODE == 1) { 
    for (int i=0; i<STRIP_SIZE; i=i+(BLOCK_SIZE_X)) {
      for (int j=0; j<FOOTPRINT_SIZE_X; j=j+(BLOCK_SIZE_X)) {
        Csub[(thread_row + i) * C.stride + (thread_col+j)] = Cvalues[d];
        d=d+1;
      }
    }
  }

  if (MODE == 2) {
    for(int i=0; i<STRIP_SIZE; i=i+(SCALING_FACTOR_Y*num_threads)) {
      for (int j=0; j<FOOTPRINT_SIZE_X; j=j+(SCALING_FACTOR_X*num_threads)) {
        for (int l=0; l<SCALING_FACTOR_X;++l) {
          for (int k=0; k<SCALING_FACTOR_Y;++k) {
            Csub[(thread_row*SCALING_FACTOR_Y + i + l)*C.stride+(thread_col*SCALING_FACTOR_X+j)+k]=Cvalues[d];
            d=d+1;
          }
        }
      }
    }
  }

}
