#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <immintrin.h>

struct timeval time;
double elapsed_time;
int posix_memalign(void **memptr, size_t alignment, size_t size);

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

void kernel(float *restrict x, float *restrict y, float *restrict z)
{
  int i,j,k;
  __m256 p_x, p_y, p_z, p_result;
  for (i=0; i<8*N; i++) 
    for (j=0; j<8*N; j++) 
      for (k=0; k<8*N; k+=8) {
        // z[k] += x[k] * y[k];
        p_x = _mm256_load_ps(&x[k]);
        p_y = _mm256_load_ps(&y[k]);
        p_z = _mm256_load_ps(&z[k]);
        p_result = _mm256_fmadd_ps(p_x, p_y, p_z);
        _mm256_store_ps(&z[k], p_result);
      }
}

void main(int argc, char *argv[])
{
  // x, y, & z should all fit in maxline L1d cache (32K bytes) if N < 312
  float *x = (float*)xmalloc(sizeof(float) * 8*N); // 32N KB
  float *y = (float*)xmalloc(sizeof(float) * 8*N); // 32N KB 
  float *z = (float*)xmalloc(sizeof(float) * 8*N); // 32N KB 

  start_timer();

  kernel(x, y, z);

  stop_timer();
  
  // prevent dead code elimination
  if (atoi(argv[0]) == 9999999) {
    for (int i=0; i<8*N; i++) 
      printf("%f,%f,%f\n",x[i],y[i],z[i]);
  }

  printf("%f\n", elapsed_time);

}
