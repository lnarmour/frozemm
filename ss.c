// MEM_LOAD_UOPS_RETIRED.L1_HIT      5301d1
// MEM_LOAD_UOPS_RETIRED.L2_HIT      5302d1
// MEM_LOAD_UOPS_RETIRED.L3_HIT      5304d1
// MEM_LOAD_UOPS_RETIRED.L1_MISS     5308d1
// MEM_LOAD_UOPS_RETIRED.L2_MISS     5310d1
// MEM_LOAD_UOPS_RETIRED.L3_MISS     5320d1
// MEM_LOAD_UOPS_RETIRED.HIT_LFB     5340d1
// L2_TRANS.ALL_PF                   5308f0
// L2_RQSTS.ALL_PF                   53f824
// L2_RQSTS.PF_MISS                  533824
// L2_RQSTS.PF_HIT                   53d824
// L2_RQSTS.ALL_DEMAND_DATA_RD       53e124
//
// perf stat -e r5301d1,r5302d1,r5304d1,r5308d1,r5310d1,r5320d1,r5340d1,r5308f0,r53f824,r533824,r53d824,r53e124 ...
//


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include "ss.h"

void MM(PRECISION alpha, PRECISION beta, 
     long N, long TSI, long TSJ, long TSK, 
     PRECISION* restrict A, 
     PRECISION* restrict B, 
     PRECISION* restrict R) 
{

	long i,j,k,ti,tj,tk;

  // #pragma omp parallel for private(tj,i,j) 
  for (ti=0; ti<N; ti+=TSI)
  for (tj=0; tj<N; tj+=TSJ)
    for (i=ti; i<min(N,ti+TSI); i++)
    for (j=tj; j<min(N,tj+TSJ); j++)
        R[i*N+j] *= beta;

  // #pragma omp parallel for private(tk,tj,i,k,j) 
  for (ti=0; ti<N; ti+=TSI)
  for (tk=0; tk<N; tk+=TSK)
  for (tj=0; tj<N; tj+=TSJ)
    for (i=ti; i<min(N,ti+TSI); i++)
    for (k=tk; k<min(N,tk+TSK); k++)
    for (j=tj; j<min(N,tj+TSJ); j++)
      R[i*N+j] += alpha * A[i*N+k] * B[k*N+j];

}


