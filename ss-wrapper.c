#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include "ss.h"

#define gflops(N, elapsed_time) 2*(N)*(N)*(N)/(elapsed_time)/1000000000
#define abs(val) (val)>0.0 ? (val) : -1*(val)

void printTile(PRECISION*, long, long);

int main(int argc, char** argv) {
  if (argc <= 1) {
    printf("Number of argument is smaller than expected.\n");
    printf("Expecting N\n");
    exit(1);
  }

  long tts1, tts2, tts3;

  long N = atoi(argv[1]);
  if (argc == 2) {
    tts1 = N;
    tts2 = N;
    tts3 = N;
  } else {
  	tts1 = atoi(argv[2]);
	  tts2 = atoi(argv[3]);
	  tts3 = atoi(argv[4]);
  }

	if (N<tts1 || N<tts2 || N<tts3) { printf("TS1, TS2, and TS3 must be less than or equal to N\n"); exit(1); }
	if (N%tts1!=0 || N%tts2!=0 || N%tts3!=0) { printf("TS1, TS2, and TS3 must divide N evenly\n"); exit(1); }

	//Timing
	struct timeval time;
	double elapsed_time;
	double times[3];
	times[0] = 0;
	times[1] = 0;
	times[2] = 0;

  gettimeofday(&time, NULL);
  elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);
	PRECISION *A = malloc(N * N * sizeof(PRECISION));
	mallocCheck(A, N*N, PRECISION);
	PRECISION *B = malloc(N * N * sizeof(PRECISION));
	mallocCheck(B, N*N, PRECISION);
	PRECISION *C = malloc(N * N * sizeof(PRECISION));
	mallocCheck(C, N*N, PRECISION);
	for (long i=0; i<N; i++)
		for (long j=0; j<N; j++) {
			*(A+i*N+j) = (PRECISION) ((i+j) % tts1);
			*(B+i*N+j) = (PRECISION) ((i+j+1) % tts3);
			*(C+i*N+j) = 0;
		}
  gettimeofday(&time, NULL);
  elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000) - elapsed_time;

  //printf("Allocation/initialization time : %lf sec.\n", elapsed_time);


#ifdef CHECK
	#define C(i,j) C[(i)*N + (j)]
	#define Check(i,j) Check[(i)*N + (j)]
	PRECISION *Check = malloc(N * N * sizeof(PRECISION));
	mallocCheck(Check, N*N, PRECISION);
	MM_MKL(ALPHA, BETA, N, N, N, A, B, Check);
#endif


  //Call the main computation
  MM(ALPHA, BETA, N, tts1, tts2, tts3, A, B, C, times);
  printf("%f\n", times[0]+times[1]+times[2]);


#ifdef CHECK
	for (long i=0; i<N; i++) {
		for (long j=0; j<N; j++) {
			PRECISION delta = abs(C(i,j)-Check(i,j));
			if (delta >= 0.001) {
				printf("Error:\n");
				printf("C(%d,%d)=%f\n", i, j, C(i,j));
				printf("Oracle(%d,%d)=%f\n", i, j, Check(i,j));
				return 1;
			}
		}
	}
#endif

}
