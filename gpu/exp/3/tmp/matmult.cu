///
/// matmult.cu
/// For CSU CS575 Spring 2011
/// Instructor: Wim Bohm
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-19 DVN
///
/// Do not modify this file. The GTA will grade your
/// code using the master copy of this file, not your
/// copy, so any modifications you make will not play
/// a role in the grading.
///

// Includes
#include <stdio.h>
#include "timer.h"
#include "matmultKernel.h"

// Defines
#define epsilon (float)1e-4
#define max(x, y)   ((x)>(y) ? (x) : (y))
#define CUDA_CHECK_RETURN(value){         \
cudaError_t _m_cudaStat = value;            \
if (_m_cudaStat != cudaSuccess){           \
fprintf(stderr, "Error: %s at line %d in file %s \n", cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);  \
exit(1); \
}}

int verbose = std::getenv("VERBOSE")!=NULL ? atoi(std::getenv("VERBOSE")) : 0;

Matrix MakeDeviceMatrix(Matrix M, bool copy){
  // Create a new matrix in device memory.
  Matrix newDeviceMatrix;
  newDeviceMatrix.width = M.width;
  newDeviceMatrix.stride = M.width;
  newDeviceMatrix.height = M.height;
  size_t size = M.width * M.height * sizeof(float);
  cudaMalloc((void**) &newDeviceMatrix.elements, size);
  if (copy)
    cudaMemcpy(newDeviceMatrix.elements, M.elements, size, cudaMemcpyHostToDevice);
  return newDeviceMatrix;
}

// Host code for matrix multiplication.
// Matrix dimensions must be multiples of size 
// This code assumes that the matrix is square.
void MatMul(const Matrix A, const Matrix B, Matrix C, int dimension1, int dimension2, int bx, int by, int tx, int ty, int cv){

  // Create device data structures.
  Matrix device_A = MakeDeviceMatrix(A, true);
  Matrix device_B = MakeDeviceMatrix(B, true);
  Matrix device_C = MakeDeviceMatrix(C, false);

  // Define grid topology
 dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
 dim3 dimGrid(B.width/FOOTPRINT_SIZE_X, A.height/FOOTPRINT_SIZE_Y);

  // Invoke kernel for warm up
  //MatMulKernel<<<dimGrid, dimBlock>>>(device_A, device_B, device_C);

  // Synchronize to make sure everyone is done in the warmup.
  cudaThreadSynchronize();
  CUDA_CHECK_RETURN(cudaGetLastError());

  // Set up timer
  initialize_timer();
  start_timer();


  // Invoke kernel for real
  MatMulKernel<<<dimGrid, dimBlock>>>(device_A, device_B, device_C, bx, by, tx, ty, cv);
 
  // Synchronize to make sure everyone is done.
  cudaThreadSynchronize() ;
  CUDA_CHECK_RETURN(cudaGetLastError());

  // Compute and report the timing results

  stop_timer();
  double time = elapsed_time();

  double nFlops = (double)A.width*A.height*B.width*2;
  double nFlopsPerSec = nFlops/time;
  double nGFlopsPerSec = nFlopsPerSec*1e-9;
  printf( "Data dimensions: %dx%d \n", C.height, C.width);
  printf( "Grid Dimensions: %dx%d \n",dimGrid.x,dimGrid.y);
  printf( "Block Dimensions: %dx%d \n",dimBlock.x,dimBlock.y);
  printf( "Footprint Dimensions: %dx%d \n",FOOTPRINT_SIZE_X,FOOTPRINT_SIZE_Y);
  
  printf( "Time: %lf (sec), nFlops: %0.0lf, GFlopsS: %lf\n",
            time, nFlops, nGFlopsPerSec);

  // Copy the result to the host memory from device memory
  size_t size = C.width * C.height * sizeof(float);
  cudaMemcpy(C.elements, device_C.elements, size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(device_A.elements);
  cudaFree(device_B.elements);
  cudaFree(device_C.elements);
   
}


// Create a matrix in host memory.
Matrix MakeHostMatrix(int width, int height){
  Matrix newHostMatrix;
  newHostMatrix.width = width;
  newHostMatrix.height = height;
  size_t size = newHostMatrix.width * newHostMatrix.height * sizeof(float);
  newHostMatrix.elements = (float*)malloc(size);
  return newHostMatrix;
}

void printMatrix(Matrix M, const char* name) {
  printf("\n%s \n",name);
  for(int y=0; y<M.height; y++){
   for(int x=0; x<M.width; x++) {
      printf("%8.2f ", M.elements[y * M.width + x]);
   }
   printf("\n");
  }
}

void initMatrix(Matrix M, bool horizontal) {
  for(int i=0; i<M.height; i++) {
    for(int j=0; j<M.width; j++) {
      if (horizontal)
        M.elements[i*M.width+j] = (float)(i+j);
      else
        M.elements[j*M.height+i] = (float)(i+j);
    }
  }
}

void checkResult(Matrix A, Matrix B, Matrix C) {

  Matrix O = MakeHostMatrix(B.width, A.height);

  for (int i=0; i<A.height; i++)
    for (int j=0; j<B.width; j++) {
      O.elements[i*O.width+j] = 0.0;
      for (int k=0; k<A.width; k++)
        O.elements[i*O.width+j] += A.elements[k*A.width+i] * B.elements[k*B.width+j];
    }

  if(verbose){
   printMatrix(C, "host_C");
   printMatrix(O, "oracle");
  }

  double maxerror = 0.0;
  int errCnt = 0;
  for (int i=0; i<O.height; i++)
    for (int j=0; j<O.width; j++) {
      float diff = O.elements[i*O.width+j] - C.elements[i*C.width+j];
      if (fabs(diff) > epsilon) {
        errCnt++;
        maxerror = max(maxerror, diff);
      }
    }  

  if(errCnt>0){
    printf("\n\nTEST FAILED: number of errors:  %d, max rel error: %f\n", errCnt, maxerror);
  }
  
  free(O.elements);
}

//
// main
//
int main(int argc, char** argv) {

  // Grid dimension
  int num_blocks;
  // Matrix dimensions in multiples of FOOTPRINT_SIZE
  // Matrices will be of size data_size * data_size
  int data_size1;
  int data_size2;
  //int mode;
  //int FOOTPRINT_SIZE;

  int bx = 0;
  int by = 0;
  int tx = 0;
  int ty = 0;
  int cv = 0;

  // Read command line argument
  if(1 < argc){
    sscanf(argv[1], "%d", &num_blocks);
    data_size1 = num_blocks * FOOTPRINT_SIZE_X;
    data_size2 = num_blocks * FOOTPRINT_SIZE_Y;
    if (6 < argc) {
      bx = atoi(argv[2]); 
      by = atoi(argv[3]); 
      tx = atoi(argv[4]); 
      ty = atoi(argv[5]); 
      cv = atoi(argv[6]); 
    }
  } else {
     printf("Usage: %s NumBlocks\n", argv[0]);
     exit(0);
  }     

  // Create matrices in host.
  Matrix host_A = MakeHostMatrix(data_size2, data_size1);
  Matrix host_B = MakeHostMatrix(data_size2, data_size1);
  Matrix host_C = MakeHostMatrix(data_size2, data_size1);

  // Initialize values in host A and B
  initMatrix(host_A,false);
  initMatrix(host_B,true);

  // debugging
  if(verbose){
    printMatrix(host_A, "host_A");
    printMatrix(host_B, "host_B");
  }

  // Perform CUDA matrix Multiplication
  // MatMul is a host function that calls
  // the device kernel MatMulKernel and
  // times its performance.
  MatMul(host_A,host_B,host_C,FOOTPRINT_SIZE_X,FOOTPRINT_SIZE_Y, bx, by, tx, ty, cv);

  // Verify that the result is correct.
  checkResult(host_A, host_B, host_C);
  
  // Free allocated memory.
  free(host_A.elements);
  free(host_B.elements);
  free(host_C.elements);
}

