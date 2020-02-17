#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "ss.h"



int main(int argc, char** argv) {
  if (argc <= 4) {
    printf("Number of argument is smaller than expected.\n");
    printf("Expecting N TSI TSJ TSK\n");
    exit(1);
  }

  long N = atoi(argv[1]);
  long tts1 = atoi(argv[2]);
  long tts2 = atoi(argv[3]);
  long tts3 = atoi(argv[4]);

  if (N<tts1 || N<tts2 || N<tts3) { printf("TS1, TS2, and TS3 must be less than or equal to N\n"); exit(1); }
  if (N%tts1!=0 || N%tts2!=0 || N%tts3!=0) { printf("TS1, TS2, and TS3 must divide N evenly\n"); exit(1); }

	long TSI = tts1;
	long TSJ = tts2;
	long TSK = tts3;
	long ti, tj, tk;

  //Timing
  struct timeval time;
  double elapsed_time;
  double times[3];

	start_timer(0);
  PRECISION *A = malloc(N * N * sizeof(PRECISION));
  PRECISION *B = malloc(N * N * sizeof(PRECISION));
  PRECISION *C = malloc(N * N * sizeof(PRECISION));
	PRECISION* scratch = (PRECISION*)malloc(sizeof(PRECISION)*N*max(TSI,TSK));
  for (long i=0; i<N; i++) {
    for (long j=0; j<N; j++) 
      *(A+i*N+j) = i*N+j; //(PRECISION)rand() / (PRECISION)RAND_MAX;
    for (long j=0; j<N; j++) 
      *(B+i*N+j) = i*N+j; //(PRECISION)rand() / (PRECISION)RAND_MAX;
    for (long j=0; j<N; j++) 
      *(C+i*N+j) = 0;
	}
	stop_timer(0);
  printf("Allocation/initialization time : %lf sec.\n", times[0]);

	start_timer(1);
	for(ti=0; ti<N/TSI; ti++) two2four(A, scratch, N, N, TSI, TSK, ti);
	for(tk=0; tk<N/TSK; tk++) two2four(B, scratch, N, N, TSK, TSJ, tk);
	stop_timer(1);
  printf("two2four time : %lf sec.\n", times[1]);


}
