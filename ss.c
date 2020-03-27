#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

struct timeval time;
double elapsed_time;
int posix_memalign(void **memptr, size_t alignment, size_t size);

#define min(x, y)   ((x)>(y) ? (y) : (x))
#define start_timer() gettimeofday(&time, NULL); elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000)
#define stop_timer() gettimeofday(&time, NULL); elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000) - elapsed_time


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


void MM(long N,
        float* restrict x0, float* restrict y0, float* restrict z0,
        float* restrict x1, float* restrict y1, float* restrict z1)
{
  long i,j,k,ti,tj,tk;

  for (ti=0; ti<N; ti+=TI)
    for (tk=0; tk<N; tk+=TK)
      for (tj=0; tj<N; tj+=TJ) {

        if (ti+TI<N && tj+TJ<N && tk+TK<N) {
          for (i=ti; i<ti+TI; i++)
            for (k=tk; k<tk+TK; k++) {
              #pragma vector aligned
              for (j=tj; j<tj+TJ; j++) {
                z0[i*N+j] += x0[i*N+k] * y0[k*N+j];
//                z0[i*N+j] += x0[i*N+k] * y1[k*N+j];
              }
            }
        } else {
          for (i=ti; i<min(N,ti+TI); i++)
            for (k=tk; k<min(N,tk+TK); k++) {
              #pragma vector aligned
              for (j=tj; j<min(N,tj+TJ); j++) {
                z0[i*N+j] += x0[i*N+k] * y0[k*N+j];
//                z0[i*N+j] += x0[i*N+k] * y1[k*N+j];
              }
            }
        }

      }

}


int main(int argc, char** argv) {

  if (argc <= 1) exit(1);
  long N = atoi(argv[1]);

	float *A = xmalloc(N * N * sizeof(float));
	float *B = xmalloc(N * N * sizeof(float));
	float *C = xmalloc(N * N * sizeof(float));
	float *D = xmalloc(N * N * sizeof(float));
	float *E = xmalloc(N * N * sizeof(float));
	float *F = xmalloc(N * N * sizeof(float));
	// float *Binv = xmalloc(N * N * sizeof(float));

	for (long i=0; i<N; i++)
		for (long j=0; j<N; j++) {
			C[i*N+j] = (float) ((i*j+1) % N) / N;
			A[i*N+j] = (float) (i*(j+1) % N) / N;
			B[i*N+j] = (float) (i*(j+2) % N) / N;
			F[i*N+j] = (float) ((i*j+1) % N) / N;
			D[i*N+j] = (float) (i*(j+1) % N) / N;
			E[i*N+j] = (float) (i*(j+2) % N) / N;
		}

  // long i,j;
  // for (i=0; i<N; i++)
  //   for (j=0; j<N; j++)  
  //     Binv[j*N+i] = B[i*N+j];

  start_timer();
  MM(N, A, B, C, D, E, F);
  stop_timer();

  printf("%f\n", elapsed_time);

}
