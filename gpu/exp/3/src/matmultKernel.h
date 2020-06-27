#ifndef __MMKERNEL__
#define __MMKERNEL__

#define BLOCK_SIZE_X 4
#define BLOCK_SIZE_Y 4

#define FOOTPRINT_SIZE_X 64
#define FOOTPRINT_SIZE_Y 64

// SCALING_FACTOR_* must be FOOTPRINT_SIZE_* / BLOCK_SIZE_*
#define SCALING_FACTOR_X 16
#define SCALING_FACTOR_Y 16

// STRIP_SIZE must divide FOOTPRINT_SIZE_X and FOOTPRINT_SIZE_Y
#define STRIP_SIZE 4

typedef struct {
  int width;
  int height;
  int stride;
  float* elements;
} Matrix;

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

#endif

