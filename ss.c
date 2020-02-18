// perf stat -e page-faults:u,major-faults:u,minor-faults:u ./MM 10000 5000 5000 5000

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl.h>
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

void MM(long N, long TSI, long TSJ, long TSK, PRECISION* A, PRECISION* B, PRECISION* R, double times[3]) {

	struct timeval time;
	long i,j,k,ti,tj,tk;

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

	start_timer(1);
	// copy first blocks of A and B into a and b

	#define A(tl,tm,TSL,TSM) A[(tl)*(N)*(TSL) + (tm)*(TSL)*(TSM)]	
	#define B(tl,tm,TSL,TSM) B[(tl)*(N)*(TSL) + (tm)*(TSL)*(TSM)]	
	#define R(tl,tm,TSL,TSM) R[(tl)*(N)*(TSL) + (tm)*(TSL)*(TSM)]	

	long ti_max = N/TSI;
	long tj_max = N/TSJ;
	long tk_max = N/TSK;
	for (ti=0; ti<ti_max; ti++) 
	for (tj=0; tj<tj_max; tj++) {
		fetch_tile(&A(ti,0,TSI,TSK), a_curr, TSI*TSK);
		fetch_tile(&B(0,tj,TSK,TSJ), b_curr, TSK*TSJ);
		for (tk=0; tk<tk_max; tk++) {
			#pragma omp parallel num_threads(2) firstprivate(ti,tj,tk,tk_max)
			{
				if (omp_get_thread_num() == 0) {
					if (tk+1<tk_max) {
						fetch_tile(&A(ti,tk+1,TSI,TSK), a_next, TSI*TSK);
						fetch_tile(&B(tk+1,tj,TSK,TSJ), b_next, TSK*TSJ);
					}
				} else {
					MM_MKL(TSI, TSK, TSJ, a_curr, b_curr, &R(ti,tj,TSI,TSJ));
				}
			}
			tmp = a_curr; a_curr = a_next; a_next = tmp;
			tmp = b_curr; b_curr = b_next; b_next = tmp;
		}
	}
	stop_timer(1);

	start_timer(2);
	four2two(A, scratch, N, N, TSI, TSK);
	four2two(B, scratch, N, N, TSK, TSJ);
	four2two(R, scratch, N, N, TSI, TSJ);
	stop_timer(2);

}


