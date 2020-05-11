#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>


#ifdef INT
#define PRECISION long
#define CBLAS_GEMM cblas_sgemm
#endif

#ifdef SINGLE
#define PRECISION float
#define CBLAS_GEMM cblas_sgemm
#endif

#ifdef DOUBLE
#define PRECISION double
#define CBLAS_GEMM cblas_dgemm
#endif

#define start_timer(i) gettimeofday(&time, NULL); times[(i)] = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000)
#define stop_timer(i) gettimeofday(&time, NULL); times[(i)] = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000) - times[(i)]

#define min(x, y)   ((x)>(y) ? (y) : (x))
#define max(x, y)   ((x)>(y) ? (x) : (y))
#define mallocCheck(v,s,d) if ((v) == NULL) { printf("Failed to allocate memory for %s : size=%lu\n", "sizeof(d)*(s)", sizeof(d)*(s)); exit(-1); }

void MM_MKL(long, long, long, PRECISION*, PRECISION*, PRECISION*);
void MM(long, long, long, long, PRECISION*, PRECISION*, PRECISION*, double[3]);
void two2four(PRECISION*, PRECISION*, long, long, long, long);
void four2two(PRECISION*, PRECISION*, long, long, long, long);
void fetch_tile(PRECISION*, PRECISION*, long);
int posix_memalign(void **memptr, size_t alignment, size_t size);

static void * xmalloc (size_t num)
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

