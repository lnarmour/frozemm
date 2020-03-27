#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl.h>

struct timeval time;
double elapsed_time;

int check(long, float*, float*, float*);
int posix_memalign(void**, size_t, size_t);

#define min(x, y) ((x)>(y) ? (y) : (x))
#define start_timer() gettimeofday(&time, NULL); elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000)
#define stop_timer() gettimeofday(&time, NULL); elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000) - elapsed_time


extern void* xmalloc (size_t num)
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


void kernel(long N, long PI, long PJ, long TK, float *A, float *B, float*C)
{
  long pi,pj,tk;
  float *a, *b, *c;

  // https://software.intel.com/en-us/forums/intel-math-kernel-library/topic/514595
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,3,3,3,1,A,6,B,6,0,C,6);
  return;

  // outer two loops to iterate over (square) patches of C
  for (pi=0; pi<N; pi+=PI)
    for (pj=0; pj<N; pj+=PJ) {
      c = &(C[pi*N+pj]);
      for (tk=0; tk<N; tk+=TK) {
        a = &(A[pi*N+tk]);
        b = &(B[tk*N+pj]);
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,PI,PJ,TK,1,a,N,b,N,0,c,N);
      }
    }
}


int main(int argc, char** argv)
{
  if (argc <= 4) exit(1);
  long N = atoi(argv[1]);
  long TI = atoi(argv[2]);
  long TJ = atoi(argv[3]);
  long TK = atoi(argv[4]);

	float *A = xmalloc(N * N * sizeof(float));
	float *B = xmalloc(N * N * sizeof(float));
	float *C = xmalloc(N * N * sizeof(float));
	// float *Binv = xmalloc(N * N * sizeof(float));

	for (long i=0; i<N; i++)
		for (long j=0; j<N; j++) {
			C[i*N+j] = (float) ((i*j+1) % N) / N;
			A[i*N+j] = (float) (i*(j+1) % N) / N;
			B[i*N+j] = (float) (i*(j+2) % N) / N;
		}

  // long i,j;
  // for (i=0; i<N; i++)
  //   for (j=0; j<N; j++)  
  //     Binv[j*N+i] = B[i*N+j];

  start_timer();
  kernel(N,TI,TJ,TK,A,B,C);
  stop_timer();

  printf("%f\n", elapsed_time);

  #ifdef CHECK
  check(N,A,B,C);
  #endif

}
