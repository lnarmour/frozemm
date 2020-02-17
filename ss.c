// perf stat -e page-faults:u,major-faults:u,minor-faults:u ./MM 10000 5000 5000 5000

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl.h>
#include "ss.h"

#define A(tl,tm,TSL,TSM,L,M) A[tl*(M*TSL) + tm*(TSL*TSM)]
#define B(tl,tm,TSL,TSM,L,M) B[tl*(M*TSL) + tm*(TSL*TSM)]
#define R(tl,tm,TSL,TSM,L,M) R[tl*(M*TSL) + tm*(TSL*TSM)]

void MM(long N, long TSI, long TSJ, long TSK, PRECISION* A, PRECISION* B, PRECISION* R, double times[3]) {

	struct timeval time;
	long i,j,k,ti,tj,tk;

	PRECISION* scratch = (PRECISION*)malloc(sizeof(PRECISION)*N*max(TSI,TSK));

	start_timer(0);
	// Execution time 0
	for (ti=0; ti<N/TSI; ti++) two2four(A, scratch, N, N, TSI, TSK, ti);
	for (tk=0; tk<N/TSK; tk++) two2four(B, scratch, N, N, TSK, TSJ, tk);
	stop_timer(0);
	
	start_timer(1);
	// Execution time 1
	for (ti=0; ti<N/TSI; ti++) {
		for (tk=0; tk<N/TSK; tk++) {
			for (tj=0; tj<N/TSJ; tj++) {
				// can we ensure no page faults occur for tj+1 tile of B and R?
				MM_MKL(TSI, TSK, TSJ, &A(ti,tk,TSI,TSK,N,N), &B(tk,tj,TSK,TSJ,N,N), &R(ti,tj,TSI,TSJ,N,N));
			}
		}
	}
	stop_timer(1);

	start_timer(2);
	// Execution time 2
	if (N!=TSI || N!=TSJ || N!=TSK) {
		for (ti=0; ti<N/TSI; ti++) four2two(A, scratch, N, N, TSI, TSK, ti);
		for (ti=0; ti<N/TSI; ti++) four2two(R, scratch, N, N, TSI, TSJ, ti);
		for (tk=0; tk<N/TSK; tk++) four2two(B, scratch, N, N, TSK, TSJ, tk);
	}
	stop_timer(2);
}


