/// matmultKernel.h
/// For CSU CS575 Spring 2011
/// Instructor: Wim Bohm
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-02-16
/// Last Modified: 2011-02-19 DVN
///
/// Kernels defined with this header must 
/// multiply two matrices using CUDA: A x B = C
///

#ifndef __MMKERNEL__
#define __MMKERNEL__

// Defines the size of thread block
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

// Defines the number of elements computed by each thread block
#ifndef FOOTPRINT_SIZE
#define FOOTPRINT_SIZE BLOCK_SIZE
#endif

// Defines the number of points that each thread will compute
#ifndef P
#define P (FOOTPRINT_SIZE * FOOTPRINT_SIZE / BLOCK_SIZE / BLOCK_SIZE)
#endif

// The type Matrix is really a MATRIX DESCRIPTOR. 
// Matrices are stored in row major order:
//       M[row,col] = *(M.elements + row * M.stride + col)
//
// A sub matrix is not copied but allocated in the full matrix.
//
// This requires the stride of the full matrix to properly get to the
// next row of the sub matrix (a block).
//
// Stride is the width in bytes from one element of the larger matrix 
// to the element in the same column but one row down.


typedef struct {
  int width;
  int height;
  int stride;
  float* elements;
} Matrix;

// Forward declaration of the kernel function that performs the work.
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

#endif

