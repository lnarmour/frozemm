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
#include <stdio.h>

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
  float *Asub, *Bsub, *Csub;
  int num_threads=BLOCK_SIZE_X*BLOCK_SIZE_Y;
  Csub = &C.elements[C.stride * STRIP_SIZE * blockIdx.y + FOOTPRINT_SIZE_X * blockIdx.x];

  // Each thread computes one element of Csub in its copy of CValue
  //int total_values = (FOOTPRINT_SIZE/BLOCK_SIZE)*(FOOTPRINT_SIZE/BLOCK_SIZE);
  float Cvalues[SCALING_FACTOR] = {[0 ... SCALING_FACTOR-1]=0};
  //for (int c=0; c < SCALING_FACTOR1; ++c){
    //     Cvalues[c]=0;      
  // }
 
  // Loop over all sub matrices in blockIdx.y of A and blockIdx.x of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
  for (int m = 0;  m < (A.width / STRIP_SIZE); ++m){
    // Get Asub and Bsub descriptors
    Asub = &A.elements[A.stride * STRIP_SIZE * m + FOOTPRINT_SIZE_Y * blockIdx.y];
    Bsub = &B.elements[B.stride * STRIP_SIZE * m + FOOTPRINT_SIZE_X * blockIdx.x];
    // Copy ELEMENTS OF  ASub and Bsub into shared memory
    // EACH THREAD loads ONE ELEMENT of ASub and ONE of Bsub
    // Notice: it does not need to be the element it requires to
    //         compute its Cvalue, as long as all elements are 
    //         collaboratively read. 

    // Notice: every thread declares shared_A and shared_B in shared memory
    //         even though a thread block has only one shared_A and one shared_B
    __shared__ float shared_A[FOOTPRINT_SIZE_Y][STRIP_SIZE];
    __shared__ float shared_B[STRIP_SIZE][FOOTPRINT_SIZE_X];
    
    for (int i=0; i<STRIP_SIZE; i=i+BLOCK_SIZE_Y){
	for (int j=0; j<FOOTPRINT_SIZE_X; j = j + BLOCK_SIZE_X){
		shared_A[threadIdx.x + j][threadIdx.y + i]=Asub[(threadIdx.y + i)*A.stride + (threadIdx.x + j)];
		shared_B[threadIdx.y + i][threadIdx.x + j]=Bsub[(threadIdx.y + i)*B.stride + (threadIdx.x + j)];
	}
     }
    __syncthreads();

     // Synchronize to ensure all elements are read
    
    // Do an inproduct of one row of shared_A and one col of shared_B
    // computing one Cvalue by accumulation
    if (MODE == 0){
	int c=0;
	#pragma unroll
	for (int i=0; i<STRIP_SIZE; i=i+(BLOCK_SIZE_Y*num_threads)){
		for (int j=0; j<FOOTPRINT_SIZE_X; j=j+(BLOCK_SIZE_X*num_threads)){
			for (int l=0; l<BLOCK_SIZE_Y; ++l){
				for (int k=0; k<BLOCK_SIZE_X; ++k){
					for (int e=0; e<STRIP_SIZE; ++e){
						Cvalues[c] += shared_A[(threadIdx.y * BLOCK_SIZE_Y)+i+l][e] * shared_B[e][(threadIdx.x * BLOCK_SIZE_X)+j+k];
					}
					c=c+1;
				}
			}
		}
	}
	__syncthreads();
    }
    if (MODE == 1){
	int c=0;
	#pragma unroll
   	for (int i=0; i<STRIP_SIZE; i=i+(BLOCK_SIZE_X)){
        	for (int j=0; j<FOOTPRINT_SIZE_X; j=j+(BLOCK_SIZE_X)){
            		for (int e=0; e<STRIP_SIZE; ++e){
                		Cvalues[c] += shared_A[threadIdx.y + i][e] * shared_B[e][threadIdx.x + j];
			}
			c=c+1;
		}
	}
	__syncthreads();
     }
     if (MODE == 2){
	int c=0;
	#pragma unroll
	for (int i=0; i<STRIP_SIZE; i=i+(SCALING_FACTOR_Y*num_threads)){
		for (int j=0; j<FOOTPRINT_SIZE_X; j=j+(SCALING_FACTOR_X*num_threads)){
			for (int l=0; l<SCALING_FACTOR_X; ++l){
				for (int k=0; k<SCALING_FACTOR_Y; ++k){
					for (int e=0; e <STRIP_SIZE; ++e){
						Cvalues[c] += shared_A[threadIdx.y*SCALING_FACTOR_Y + i + l][e] * shared_B[e][threadIdx.x*SCALING_FACTOR_X + j + k];
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
    if (MODE == 0){
	for (int i=0; i<STRIP_SIZE; i=i+(BLOCK_SIZE_Y*num_threads)){
		for (int j=0; j<FOOTPRINT_SIZE_X; j=j+(BLOCK_SIZE_X*num_threads)){
			for (int l=0; l<BLOCK_SIZE_Y; ++l){
				for (int k=0; k<BLOCK_SIZE_X; ++k){
					Csub[((threadIdx.y*BLOCK_SIZE_Y)+i+l)*C.stride + (threadIdx.x*BLOCK_SIZE_X)+j+k] = Cvalues[d];
					d=d+1;
				}
			}
		}
	}
    }
    if (MODE == 1){ 
	for (int i=0; i<STRIP_SIZE; i=i+(BLOCK_SIZE_X)){
		for (int j=0; j<FOOTPRINT_SIZE_X; j=j+(BLOCK_SIZE_X)){
			Csub[(threadIdx.y + i) * C.stride + (threadIdx.x+j)] = Cvalues[d];
			d=d+1;
		}
	}
     }
     if (MODE == 2){
	for(int i=0; i<STRIP_SIZE; i=i+(SCALING_FACTOR_Y*num_threads)){
		for (int j=0; j<FOOTPRINT_SIZE_X; j=j+(SCALING_FACTOR_X*num_threads)){
			for (int l=0; l<SCALING_FACTOR_X;++l){
				for (int k=0; k<SCALING_FACTOR_Y;++k){
					Csub[(threadIdx.y*SCALING_FACTOR_Y + i + l)*C.stride+(threadIdx.x*SCALING_FACTOR_X+j)+k]=Cvalues[d];
					d=d+1;
				}
			}
		}
	}
     }
}
