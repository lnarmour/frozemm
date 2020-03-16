#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define max(x, y)   ((x)>(y) ? (x) : (y))
#define mallocCheck(v,s,d) if ((v) == NULL) { printf("Failed to allocate memory for %s : size=%lu\n", "sizeof(d)*(s)", sizeof(d)*(s)); exit(-1); }

void two2four(float* restrict I, float* restrict scratch, long L, long M, long TSL, long TSM) {
  long ti, tl, tm, l, m, i, j, u;

  const char* s = getenv("STOP");
  long STOP = atoi(s);
  if (STOP == 0 || STOP > L/TSL)
    STOP = L/TSL;

  if (L == TSL && M == TSM)
    return;

  #pragma omp parallel for
  for (ti=0; ti<STOP; ti++) {
    for (i=ti*TSL; i<(ti+1)*TSL; i++) {
      for (j=0; j<M; j++) {
        scratch[(i%TSL)*M + j] = I[i*M + j];
      }
    }
    for (i=ti*TSL; i<(ti+1)*TSL; i++) {
      #pragma vector always
      for (j=0; j<M; j++) {
        tl = i / TSL;
        tm = j / TSM;
        l = i % TSL;
        m = j % TSM;
        u = tl*(M*TSL) + tm*(TSL*TSM) + l*TSM + m;
        I[u] = scratch[(i%TSL)*M + j];
      }
    }
  }
}


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

  float *scratch = malloc(sizeof(float)*N*max(tts1,tts3));
	mallocCheck(scratch, N*max(tts1,tts3), float);
	float *A = malloc(N * N * sizeof(float));
	mallocCheck(A, N*N, float);
	float *B = malloc(N * N * sizeof(float));
	mallocCheck(B, N*N, float);
	float *C = malloc(N * N * sizeof(float));
	mallocCheck(C, N*N, float);

  printf("Total memory footprint: %f GB\n", (3.0*N*N + N*max(tts1,tts3))*sizeof(float)/1000000000);


	//Timing
	struct timeval time;
	double elapsed_time;

  gettimeofday(&time, NULL);
  elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

  //Call the main computation
  two2four(A, scratch, N, N, tts1, tts2);
  two2four(B, scratch, N, N, tts1, tts2);
  two2four(C, scratch, N, N, tts1, tts2);

  gettimeofday(&time, NULL);
  elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000) - elapsed_time;
  printf("Time : %f sec\n", elapsed_time);







  // prevent dead code optimization
  if (atoi(argv[0]) == 99999999) {
	  for (long i=0; i<N; i++) 
	  	for (long j=0; j<N; j++) {
	  		printf("%f\n", A[i*N+j]);
	  		printf("%f\n", B[i*N+j]);
	  		printf("%f\n", C[i*N+j]);
  		}
	}

}
