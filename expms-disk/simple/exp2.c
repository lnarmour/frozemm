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

#if defined (HUGE_2MB) || defined (HUGE_1GB)
  ret = madvise(new, num, MADV_HUGEPAGE);
  if (! new || ret)
    {
      fprintf (stderr, "madvise: failed");
      exit (1);
    }
#endif

  return new;
}

void init_rand_chunk_order(long *order, long num_chunks)
{
  long i,r,tmp;
  #define rando(upper) ((long)(rand()%((upper)+1)))
  for (i=0; i<num_chunks; i++)
    order[i] = i;
  for (i=num_chunks-1; i>=0; i--) {
    r = rando(i);
    tmp = order[i];
    order[i] = order[r];
    order[r] = tmp;
  }
}

void init_arr(long N, double *X, double v)
{
  for (long i=0; i<N; i++)
    X[i] = v;
}

void kernel(long N, long B, double *X, long *order, long num_chunks)
{
  long o,ti,i;
  for (o=0; o<num_chunks; o++) {
    ti = order[o] * B;
    for (i=ti; i<min(N,ti+B); i++) {
      X[i] = 2.0;
      printf("%ld %ld %ld : %ld\n", o, order[o], ti, i);
    }
  }
}


int main(int argc, char** argv)
{
  if (argc <=2) return -1;
  
  long gbs = atol(argv[1]);
  long chunk_size = atol(argv[2]);

  long size_array = gbs * (1<<30); 
  double *X = xmalloc(size_array);

  long N = size_array / sizeof(double);
  long B = chunk_size / sizeof(double);
  long num_chunks = N/B + 1;
  long *order = xmalloc(num_chunks * sizeof(long));
  init_rand_chunk_order(order, num_chunks);

  // flush X by touching every element once first
  init_arr(N,X,1.0);

  start_timer();
  kernel(N,B,X,order,num_chunks);
  stop_timer();

  printf("%f\n", elapsed_time);

  free(X);
}
