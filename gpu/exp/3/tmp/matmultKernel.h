#ifndef __MMKERNEL__
#define __MMKERNEL__

#define BLOCK_SIZE_X 2
#define BLOCK_SIZE_Y 2

#define SCALING_FACTOR_X 2
#define SCALING_FACTOR_Y 2

#define STRIP_SIZE 2
#define SCALING_FACTOR 4

#ifndef FOOTPRINT_SIZE_X
#define FOOTPRINT_SIZE_X BLOCK_SIZE_X
#endif

#ifndef FOOTPRINT_SIZE_Y
#define FOOTPRINT_SIZE_Y BLOCK_SIZE_Y
#endif

typedef struct {
  int width;
  int height;
  int stride;
  float* elements;
} Matrix;

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

#endif

