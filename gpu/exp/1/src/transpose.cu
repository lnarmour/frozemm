/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// taken & modified from:
// https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu

#include <stdio.h>
#include <assert.h>
#include "nvmlPower.hpp"
#include "cuda_profiler_api.h"

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 1000; 

__global__ void transposeNoBankConflicts(float *odata, const float *idata, int fmas_per_xfer)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  float tmp_I, tmp_O;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    tmp_I = idata[(y+j)*width + x];
    tmp_O = tmp_I;
    for (int c = 1; c <= fmas_per_xfer; c++) 
      tmp_O += c * tmp_I;
    tile[threadIdx.y+j][threadIdx.x] = tmp_O;
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) 
    odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main(int argc, char **argv)
{
  int N = 1024;
  if (argc > 1) 
    N = atoi(argv[1]);
  const int nx = N;
  const int ny = N;
  const int mem_size = nx*ny*sizeof(float);

  dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

  int devId = 0;
  int fmas_per_xfer = 0;
  if (argc > 2) 
    fmas_per_xfer = atoi(argv[2]);

  printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", 
         nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
  printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
         dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
  
  checkCuda( cudaSetDevice(devId) );

  float *h_idata = (float*)malloc(mem_size);
  float *h_cdata = (float*)malloc(mem_size);
  float *h_tdata = (float*)malloc(mem_size);
  float *gold    = (float*)malloc(mem_size);
  
  float *d_idata, *d_cdata, *d_tdata;
  checkCuda( cudaMalloc(&d_idata, mem_size) );
  checkCuda( cudaMalloc(&d_cdata, mem_size) );
  checkCuda( cudaMalloc(&d_tdata, mem_size) );

  // check parameters and calculate execution configuration
  if (nx % TILE_DIM || ny % TILE_DIM) {
    printf("nx and ny must be a multiple of TILE_DIM (%d)\n", TILE_DIM);
    goto error_exit;
  }

  if (TILE_DIM % BLOCK_ROWS) {
    printf("TILE_DIM (%d) must be a multiple of BLOCK_ROWS (%d)\n", TILE_DIM, BLOCK_ROWS);
    goto error_exit;
  }
    
  // host
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++) 
      h_idata[j*nx + i] = j*nx + i;

  // correct result for error checking
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      gold[j*nx + i] = h_idata[i*nx + j];
  
  // device
  checkCuda( cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice) );
  
  // events for timing
  cudaEvent_t startEvent, stopEvent;
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  float ms;

  // ------------
  // time kernel
  // ------------
  checkCuda( cudaMemset(d_tdata, 0, mem_size) );
  // warmup
  for (int i = 0; i < NUM_REPS; i++)
    transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata, fmas_per_xfer);
  nvmlAPIRun();
  checkCuda( cudaEventRecord(startEvent, 0) );
#ifdef NVPROFILE
  cudaProfilerStart();
  transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata, fmas_per_xfer);
  cudaProfilerStop();
#else
  for (int i = 0; i < NUM_REPS; i++)
    transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata, fmas_per_xfer);
#endif
  checkCuda( cudaEventRecord(stopEvent, 0) );
  cudaDeviceSynchronize();
  nvmlAPIEnd();  

  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  checkCuda( cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost) );
#ifdef NVPROFILE
  printf("throughput: %.2f GB/s\n", 2 * nx * ny * sizeof(float) * 1e-6 / ms );
#else
  printf("throughput: %.2f GB/s\n", 2 * nx * ny * sizeof(float) * 1e-6 * NUM_REPS / ms );
#endif
  printf("time: %.0f ms\n", ms);

error_exit:
  // cleanup
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  checkCuda( cudaFree(d_tdata) );
  checkCuda( cudaFree(d_cdata) );
  checkCuda( cudaFree(d_idata) );
  free(h_idata);
  free(h_tdata);
  free(h_cdata);
  free(gold);
}