// perf stat -e page-faults:u,major-faults:u,minor-faults:u ./MM 10000 5000 5000 5000

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl.h>
#include <omp.h>
#include "ss.h"

#define A_scratch(sel,TSL,TSM) A_scratch[((sel)%2)*(TSL)*(TSM)]
#define B_scratch(sel,tl,tm,TSL,TSM) B_scratch[((sel)%2)*(N)*(TSL) + (tm)*(TSL)*(TSM)]
#define R(tl,tm,TSL,TSM,L,M) R[tl*(M*TSL) + tm*(TSL*TSM)]

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

#ifdef PARALLEL
	PRECISION* A_scratch = (PRECISION*)xmalloc(sizeof(PRECISION)*2*TSI*TSK);
	PRECISION* B_scratch = (PRECISION*)xmalloc(sizeof(PRECISION)*2*N*TSK);
#else
	PRECISION* A_scratch = (PRECISION*)xmalloc(sizeof(PRECISION)*TSI*TSK);
	PRECISION* B_scratch = (PRECISION*)xmalloc(sizeof(PRECISION)*N*TSK);
#endif
	PRECISION* R_scratch = (PRECISION*)xmalloc(sizeof(PRECISION)*N*TSI);

	start_timer(1);
	// Execution time 1
#ifdef PARALLEL
	#pragma omp parallel
	{
		omp_set_max_active_levels(2);
		#pragma omp single
		{
			for (ti=0; ti<N/TSI; ti++) {
				two2four_single(A, &A_scratch(0,TSI,TSK), N, N, TSI, TSK, ti, 0);
				for (tk=0; tk<N/TSK; tk++) {
					if ((tk-1)<(N/TSK)) two2four_single(A, &A_scratch((tk+1),TSI,TSK), N, N, TSI, TSK, ti, tk+1);
					two2four_single(B, &B_scratch[(tk%2)*N*TSK], N, N, TSK, TSJ, tk, 0);
					for (tj=0; tj<N/TSJ; tj++) {
						#pragma omp task
						{	
							if (tj-1<N/TSJ)
								two2four_single(B, &B_scratch[(tk%2)*N*TSK + (tj+1)*TSK*TSJ], N, N, TSK, TSJ, tk, tj+1); 
						}
						MM_MKL(TSI, TSK, TSJ, &A_scratch(tk,TSI,TSK), &B_scratch[(tk%2)*N*TSK + tj*TSK*TSJ], &R(ti,tj,TSI,TSJ,N,N));
						#pragma omp taskwait
					}
				}
			}
		}
	}
#else
	for (ti=0; ti<N/TSI; ti++) {
		for (tk=0; tk<N/TSK; tk++) {
			two2four_single(A, &A_scratch(tk,TSI,TSK), N, N, TSI, TSK, ti, tk);
			for (tj=0; tj<N/TSJ; tj++) {
				two2four_single(B, &B_scratch[tj*TSK*TSJ], N, N, TSK, TSJ, tk, tj); 
				MM_MKL(TSI, TSK, TSJ, &A_scratch(tk,TSI,TSK), &B_scratch[tj*TSK*TSJ], &R(ti,tj,TSI,TSJ,N,N));
			}
		}
	}
#endif
	stop_timer(1);

	start_timer(2);
	// Execution time 2
	if (N!=TSI || N!=TSJ || N!=TSK) {
		for (ti=0; ti<N/TSI; ti++) four2two(R, R_scratch, N, N, TSI, TSJ, ti);
	}
	stop_timer(2);
}


