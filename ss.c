// perf stat -e page-faults:u,major-faults:u,minor-faults:u ./MM 10000 5000 5000 5000

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl.h>
#include <omp.h>
#include "ss.h"

#define gflops(N, elapsed_time) 2*(N)*(N)*(N)/(elapsed_time)/1000000000

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

void MM(long N, long TSI, long TSJ, long TSK, PRECISION* A, PRECISION* B, PRECISION* R, double times[3]) {

	struct timeval time;
	long i,j,k,ti,tj,tk,t;

	PRECISION* scratch = (PRECISION*)xmalloc(sizeof(PRECISION)*N*max(TSI,TSK));
	PRECISION* a_next = (PRECISION*)xmalloc(sizeof(PRECISION)*TSI*TSK);
	PRECISION* a_curr = (PRECISION*)xmalloc(sizeof(PRECISION)*TSI*TSK);
	PRECISION* b_next = (PRECISION*)xmalloc(sizeof(PRECISION)*TSK*TSJ);
	PRECISION* b_curr = (PRECISION*)xmalloc(sizeof(PRECISION)*TSK*TSJ);
	PRECISION* tmp;

	start_timer(0);
	two2four(A, scratch, N, N, TSI, TSK);
	two2four(B, scratch, N, N, TSK, TSJ);
	two2four(R, scratch, N, N, TSI, TSJ);
	stop_timer(0);
	printf("Time 0 : %lf sec.\n", times[0]);

	start_timer(1);
	#define A(tl,tm,TSL,TSM) A[(tl)*(N)*(TSL) + (tm)*(TSL)*(TSM)]	
	#define B(tl,tm,TSL,TSM) B[(tl)*(N)*(TSL) + (tm)*(TSL)*(TSM)]	
	#define R(tl,tm,TSL,TSM) R[(tl)*(N)*(TSL) + (tm)*(TSL)*(TSM)]	
	#define ti(t) ((t)/((N*N)/(TSJ*TSK)))
	#define tj(t) (((t)%((N*N)/(TSJ*TSK)))/(N/TSK))
	#define tk(t) (((t)%((N*N)/(TSJ*TSK)))%(N/TSK))

	long T = N*N*N/(TSI*TSJ*TSK);

	fetch_tile(A, a_curr, TSI*TSK);
	fetch_tile(B, b_curr, TSK*TSJ);
	
	#pragma omp parallel num_threads(2) firstprivate(t,T)
	{
		mkl_set_dynamic(0);
		omp_set_nested(1);
		omp_set_max_active_levels(2);
		for (t=0; t<T; t++) {
			if (omp_get_thread_num() == 0) {
				if (t+1<T) {
					fetch_tile(&A(ti(t+1),tk(t+1),TSI,TSK), a_next, TSI*TSK);
					fetch_tile(&B(tk(t+1),tj(t+1),TSK,TSJ), b_next, TSK*TSJ);
				}
			} else {
				MM_MKL(TSI, TSK, TSJ, a_curr, b_curr, &R(ti(t),tj(t),TSI,TSJ));
			}
			#pragma omp barrier
			#pragma omp single
			{
				tmp = a_curr; a_curr = a_next; a_next = tmp;
				tmp = b_curr; b_curr = b_next; b_next = tmp;
			}
		}
	}
	stop_timer(1);
	printf("Time 1 : %lf sec (%f glfops/sec relative).\n", times[1], gflops(N, times[1]));

	start_timer(2);
	four2two(A, scratch, N, N, TSI, TSK);
	four2two(B, scratch, N, N, TSK, TSJ);
	four2two(R, scratch, N, N, TSI, TSJ);
	stop_timer(2);
	printf("Time 2 : %lf sec.\n", times[2]);

}


