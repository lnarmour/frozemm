// perf stat -e page-faults:u,major-faults:u,minor-faults:u ./MM 10000 5000 5000 5000

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl.h>
#include "ss.h"

#define A_scratch(sel,TSL,TSM) A_scratch[((sel)%2)*(TSL)*(TSM)]
#define B_scratch(sel,tl,tm,TSL,TSM) B_scratch[((sel)%2)*(N)*(TSL) + (tm)*(TSL)*(TSM)]
#define R(tl,tm,TSL,TSM,L,M) R[tl*(M*TSL) + tm*(TSL*TSM)]

void MM(long N, long TSI, long TSJ, long TSK, PRECISION* A, PRECISION* B, PRECISION* R, double times[3]) {

	struct timeval time;
	long i,j,k,ti,tj,tk;

	PRECISION* A_scratch = (PRECISION*)malloc(sizeof(PRECISION)*2*TSI*TSK);
	PRECISION* B_scratch = (PRECISION*)malloc(sizeof(PRECISION)*2*N*TSK);
	PRECISION* R_scratch = (PRECISION*)malloc(sizeof(PRECISION)*N*TSI);

	start_timer(1);
	// Execution time 1
	for (ti=0; ti<N/TSI; ti++) {
		two2four_single(A, &A_scratch(0,TSI,TSK), N, N, TSI, TSK, ti, 0);
		for (tk=0; tk<N/TSK; tk++) {
			if ((tk-1)<(N/TSK)) two2four_single(A, &A_scratch((tk+1),TSI,TSK), N, N, TSI, TSK, ti, tk+1);
			for (tj=0; tj<N/TSJ; tj++) {
				two2four_single(B, &B_scratch(tk,tk,tj,TSK,TSJ), N, N, TSK, TSJ, tk, tj); 
				MM_MKL(TSI, TSK, TSJ, &A_scratch(tk,TSI,TSK), &B_scratch(tk,tk,tj,TSK,TSJ), &R(ti,tj,TSI,TSJ,N,N));
			}
		}
	}
	stop_timer(1);

	start_timer(2);
	// Execution time 2
	if (N!=TSI || N!=TSJ || N!=TSK) {
		for (ti=0; ti<N/TSI; ti++) four2two(R, R_scratch, N, N, TSI, TSJ, ti);
	}
	stop_timer(2);
}


