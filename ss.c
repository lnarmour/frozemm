// perf stat -e page-faults:u,major-faults:u,minor-faults:u ./MM 10000 5000 5000 5000

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl.h>
#include "ss.h"

#define A(tl,tm,TSL,TSM,L,M) A[tl*(M*TSL) + tm*(TSL*TSM)]
#define B(tl,tm,TSL,TSM,L,M) B[tl*(M*TSL) + tm*(TSL*TSM)]
#define R(tl,tm,TSL,TSM,L,M) R[tl*(M*TSL) + tm*(TSL*TSM)]

#define B_scratch(tl,tm,TSL,TSM) B_scratch[tm*(TSL*TSM)]

void MM(long N, long TSI, long TSJ, long TSK, PRECISION* A, PRECISION* B, PRECISION* R, double times[3]) {

	struct timeval time;
	long i,j,k,ti,tj,tk;

	PRECISION* A_scratch = (PRECISION*)malloc(sizeof(PRECISION)*TSI*TSK);
	PRECISION* B_scratch = (PRECISION*)malloc(sizeof(PRECISION)*N*max(TSI,TSK));

	start_timer(1);
	// Execution time 1
	for (ti=0; ti<N/TSI; ti++) {
		for (tk=0; tk<N/TSK; tk++) {

			two2four_single(A, A_scratch, N, N, TSI, TSK, ti, tk);

			for (tj=0; tj<N/TSJ; tj++) {

				two2four_row(B, B_scratch, N, N, TSK, TSJ, tk, tj);

				// can we ensure no page faults occur for tj+1 tile of B and R?
				MM_MKL(TSI, TSK, TSJ, A_scratch, &B_scratch(tk,tj,TSK,TSJ), &R(ti,tj,TSI,TSJ,N,N));
			}
		}
	}
	stop_timer(1);

	start_timer(2);
	// Execution time 2
	if (N!=TSI || N!=TSJ || N!=TSK) {
		for (ti=0; ti<N/TSI; ti++) four2two(R, B_scratch, N, N, TSI, TSJ, ti);
	}
	stop_timer(2);
}


