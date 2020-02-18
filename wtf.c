#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include "ss.h"

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


int main(int argc, char** argv) {
  if (argc <= 1) {
    printf("Number of argument is smaller than expected.\n");
    printf("Expecting N\n");
    exit(1);
  }

  long N = atoi(argv[1]);

  //Timing
  struct timeval time;
  double times[5];

  PRECISION *A = xmalloc(N * N * sizeof(PRECISION));
  PRECISION *B = xmalloc(N * N * sizeof(PRECISION));
  PRECISION *C = xmalloc(N * N * sizeof(PRECISION));
  PRECISION *X = xmalloc(N * N * sizeof(PRECISION));
  PRECISION *Y = xmalloc(N * N * sizeof(PRECISION));

  for (long i=0; i<N; i++) {
    for (long j=0; j<N; j++) {
      *(A+i*N+j) = ((i+1)*N%(N+1)+j)/N;
      *(B+i*N+j) = ((i+2)*N%(N+2)+j)/N;
      *(X+i*N+j) = ((i+3)*N%(N+3)+j)/N;
      *(Y+i*N+j) = ((i+4)*N%(N+4)+j)/N;
      *(C+i*N+j) = 0;
		}
	}

#ifdef SEQUENTIAL
	start_timer(0);
	for (long i=0; i<N; i++)
		for (long j=0; j<N; j++)
			X[i*N + j] = Y[j*N + i];
	for (long i=0; i<N; i++)
		for (long j=0; j<N; j++)
			X[j*N + i] = Y[i*N + j];
	stop_timer(0);
	start_timer(1);
	MM_MKL(N,N,N,A,B,C);
	stop_timer(1);
  printf("I/O Time : %lf sec.\n", times[0]);
  printf("Compute time : %lf sec.\n", times[1]);
#else
	start_timer(0);
	#pragma omp parallel
	{
		omp_set_max_active_levels(2);
		#pragma omp single
		{
			#pragma omp task
			{
				for (long i=0; i<N; i++)
					for (long j=0; j<N; j++)
						X[i*N + j] = Y[j*N + i];
				for (long i=0; i<N; i++)
					for (long j=0; j<N; j++)
						X[j*N + i] = Y[i*N + j];
			}
			MM_MKL(N,N,N,A,B,C);
		}
	}
	stop_timer(0);
  printf("I/O & compute time : %lf sec.\n", times[0]);
#endif

	// prevent compiler for optimizing out dead code
	if (atoi(argv[1]) == 9999999) {
		for (long i=0; i<N; i++)
			for (long j=0; j<N; j++) 
				printf("%f,%f\n", X[i*N + j], Y[i*N + j]);
	}

}
