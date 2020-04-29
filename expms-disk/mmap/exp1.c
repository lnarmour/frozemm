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
  int ret = posix_memalign (&new, 32, num);
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

double * hog_memory(long GBs)
{
  long N = GBs * (1<<30) / sizeof(double);  // 2^GBs
  double *Hog = xmalloc(N * sizeof(double));
  init_arr(N,Hog,0.0);
  return Hog;
}

double kernel(long N, long B, double *X, long *order, long num_chunks)
{
  long o,ti,i;
  double ret;
  for (o=0; o<num_chunks; o++) {
    ti = order[o] * B;
    for (i=ti; i<min(N,ti+B); i++) {
      X[i] = 0.0; // version 1 - WRITES
//      ret += X[i]; // version 2 - READS
    }
  }
  return ret;
}

int main(int argc, char** argv)
{
  if (argc <= 3) exit(1);
  long hog_GBs = atol(argv[1]);
  long measure_GBs = atol(argv[2]); 
  long size_chunks = atol(argv[3]); // size of B in bytes
  long size_array = measure_GBs * (1<<30); // size of array in bytes
#if defined (HUGE_2MB)
  long num_large_pages = size_array / (2097152UL);
	if ( size_array != num_large_pages*2097152UL ) num_large_pages++;
	size_array = (size_t) num_large_pages * 2097152UL;
  double *X = xmalloc(size_array);
#elif defined (HUGE_1GB)
  long num_large_pages = size_array / (1073741824UL);
	if ( size_array != num_large_pages*1073741824UL ) num_large_pages++;
	size_array = (size_t) num_large_pages * 1073741824UL;
  double *X = xmalloc(size_array);
#else
  double *X = xmalloc(size_array);
#endif


  long N = size_array / sizeof(double);
  long B = size_chunks / sizeof(double);
  long num_chunks = (N/B)+1;

  srand(time(0));


  long *order = xmalloc(num_chunks * sizeof(long));
  //init_arr(N,X,1.0);
  init_rand_chunk_order(order, num_chunks);  

  double *hog = hog_memory(hog_GBs);
  double ret;

  start_timer();
  ret = kernel(N,B,X,order,num_chunks);
  stop_timer();

  printf("%f\n", elapsed_time);


  if (size_array == 9999)
    printf("%f%f", hog[size_array], ret);

  free(X);
  free(hog);
}
