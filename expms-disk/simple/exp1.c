#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <time.h>

struct timeval ttime;
double elapsed_time;
int posix_memalign(void**, size_t, size_t);
int madvise(void*, size_t, int);

#define min(x, y) ((x)>(y) ? (y) : (x))
#define start_timer() gettimeofday(&ttime, NULL); elapsed_time = (((double) ttime.tv_sec) + ((double) ttime.tv_usec)/1000000)
#define stop_timer() gettimeofday(&ttime, NULL); elapsed_time = (((double) ttime.tv_sec) + ((double) ttime.tv_usec)/1000000) - elapsed_time

// took from PolyBench
extern void* xmalloc (size_t num)
{
  void* new = NULL;

#if defined (HUGE_2MB)
  int ret = posix_memalign (&new, 2097152, num);
#elif defined (HUGE_1MB)
  int ret = posix_memalign (&new, 1048576, num);
#elif defined (HUGE_1GB)
  int ret = posix_memalign (&new, 1073741824, num);
#else
  int ret = posix_memalign (&new, 4096, num);
#endif
  if (! new || ret)
    {
      fprintf (stderr, "[PolyBench] posix_memalign: cannot allocate memory");
      exit (1);
    }

#if defined (HUGE_2MB) || defined (HUGE_1GB) || defined (HUGE_1MB)
  ret = madvise(new, num, MADV_HUGEPAGE);
  if (! new || ret)
    {
      fprintf (stderr, "madvise: failed");
      exit (1);
    }
#endif


  return new;
}

void init_arr_forwards(long N, double *X, double v)
{
  for (long i=0; i<N; i++) 
    X[i] = v;
}

void init_arr_backwards(long N, double *X, double v)
{
  for (long i=0; i<N; i++) 
    X[N-1-i] = v;
}

int main(int argc, char** argv)
{
  if (argc <=1) return -1;
  
  long gbs = atol(argv[1]);
  long size_array = gbs * (1<<30); 
  long N = size_array / sizeof(double);

  double *X = xmalloc(size_array);

  init_arr_forwards(N, X, 1.0);
  
  start_timer();
#if defined (BACKWARDS)
  init_arr_backwards(N, X, 2.0);
#else // FORWARDS
  init_arr_forwards(N, X, 2.0);
#endif
  stop_timer();

  printf("%f\n", elapsed_time);

  free(X);
}
