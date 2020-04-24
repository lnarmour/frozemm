#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

struct timeval time;
double elapsed_time;
int posix_memalign(void**, size_t, size_t);

#define min(x, y) ((x)>(y) ? (y) : (x))
#define start_timer() gettimeofday(&time, NULL); elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000)
#define stop_timer() gettimeofday(&time, NULL); elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000) - elapsed_time


// took from PolyBench
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


void kernel1(long N, double *X)
{
  long i;
  for (i=0; i<N*N; i++)
    X[i] = 0.0;
}

int main(int argc, char** argv)
{
  if (argc <= 2) exit(1);
  long N = atoi(argv[1]);
  long B = atoi(argv[2]);

  double *X = xmalloc(N * N * sizeof(double));
  
  start_timer();
  kernel1(N,X);
  stop_timer();

  printf("%f\n", elapsed_time);

  #ifdef CHECK
  check(N,A,B,C);
  #endif

}
