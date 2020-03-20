#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

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

void kernel(float *restrict x0, float *restrict y0, float *restrict z0,
            float *restrict x1, float *restrict y1, float *restrict z1)
{
  int i,j,k;
  for (i=0; i<N; i++) 
    for (j=0; j<N; j++) 
      for (k=0; k<N; k++) {
        z0[k] += x0[k] * y0[k];
        z1[k] += x1[k] * y1[k];
      }
}

void main(int argc, char *argv[])
{
  float *x0 = (float*)xmalloc(sizeof(float) * N);
  float *y0 = (float*)xmalloc(sizeof(float) * N);
  float *z0 = (float*)xmalloc(sizeof(float) * N);
  float *x1 = (float*)xmalloc(sizeof(float) * N);
  float *y1 = (float*)xmalloc(sizeof(float) * N);
  float *z1 = (float*)xmalloc(sizeof(float) * N);

  start_timer();
  kernel(x0,y0,z0,x1,y1,z1);
  stop_timer();
  
  printf("%f\n", elapsed_time);
}
