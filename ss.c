// perf stat -e page-faults:u,major-faults:u,minor-faults:u ./MM 10000 5000 5000 5000

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include "ss.h"

#define gflops(N, elapsed_time, STOP) 2*(N)*(N)*(N)/(elapsed_time)/1000000000

int posix_memalign(void **memptr, size_t alignment, size_t size);
  
static void * xmalloc (size_t num)
{ 
  void* new = NULL;
  int ret = posix_memalign (&new, 32, num);
  if (! new || ret)
    {
      fprintf (stderr, "[PolyBench] posix_memalign: cannot allocate memory");
      exit (1);
    }
  return new;
}   

void MM(long N, long TSI, long TSJ, long TSK, PRECISION* restrict A, PRECISION* restrict B, PRECISION* restrict R, double times[3]) {

	struct timeval time;
	long i,j,k,ti,tj,tk;

  printf("Total memory footprint: %f Gb\n", ((3.0*N*N)*sizeof(PRECISION))/1000000000);

	start_timer(1);

  for (i=0; i<N; i++)
    for (k=0; k<N; k++)
      for (j=0; j<N; j++)
        R[i*N+j] += A[i*N+k] * B[k*N+j];

	stop_timer(1);
	printf("Time 1 : %lf sec (%f glfops/sec relative).\n", times[1], gflops(N, times[1], STOP));


}


