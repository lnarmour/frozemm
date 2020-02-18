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
	PRECISION* a_prev = (PRECISION*)xmalloc(sizeof(PRECISION)*TSI*TSK);
	PRECISION* a_curr = (PRECISION*)xmalloc(sizeof(PRECISION)*TSI*TSK);
	PRECISION* b_prev = (PRECISION*)xmalloc(sizeof(PRECISION)*TSK*TSJ);
	PRECISION* b_curr = (PRECISION*)xmalloc(sizeof(PRECISION)*TSK*TSJ);
	PRECISION* r_prev = (PRECISION*)xmalloc(sizeof(PRECISION)*TSI*TSJ);
	PRECISION* r_curr = (PRECISION*)xmalloc(sizeof(PRECISION)*TSI*TSJ);

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

	for (ti=0; ti<N/TSI; ti++) 
	for (tj=0; tj<N/TSJ; tj++) 
	for (tk=0; tk<N/TSK; tk++) {
		fetch_tile(&A(ti,tk,TSI,TSK), a_curr, TSI*TSK);
		fetch_tile(&B(tk,tj,TSK,TSJ), b_curr, TSK*TSJ);
		MM_MKL(TSI, TSK, TSJ, a_curr, b_curr, &R(ti,tj,TSI,TSJ));
	}

//	#pragma omp parallel num_threads(2)
//	{
//		int tid = omp_get_thread_num();
//		if (tid == 0) {
//			
//		} else {
//			MM_MKL(TSI, TSK, TSJ, &A, &B_scratch[tj*TSK*TSJ], &R(ti,tj,TSI,TSJ,N,N));
//		}
//
//	}
	stop_timer(1);

	start_timer(2);
	four2two(R, scratch, N, N, TSI, TSJ);
	stop_timer(2);

}


