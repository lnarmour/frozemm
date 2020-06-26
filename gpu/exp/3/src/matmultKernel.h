#ifndef __MMKERNEL__
#define __MMKERNEL__

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

#define FOOTPRINT_SIZE_X 32
#define FOOTPRINT_SIZE_Y 32

#define SCALING_FACTOR_X 2
#define SCALING_FACTOR_Y 2

#define STRIP_SIZE 16

typedef struct {
  int width;
  int height;
  int stride;
  float* elements;
} Matrix;

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

#endif

