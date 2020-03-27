#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

struct timeval time;
double elapsed_time;
int posix_memalign(void **memptr, size_t alignment, size_t size);

#define start_timer() gettimeofday(&time, NULL); elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000)
#define stop_timer() gettimeofday(&time, NULL); elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000) - elapsed_time

#ifndef N
#define N 2000
#endif

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
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k++) {
        z0[j] += x0[k] * y0[j+k];
        z0[j] += x1[k] * y1[j+k];
        z1[j+1] += x0[k] * y0[j+1+k];
        z1[j+1] += x1[k] * y1[j+1+k];
      }
    }
}


void main(int argc, char *argv[])
{
  // x, y, & z should all fit in maxline L1d cache (32K bytes) if N < 1250
  float *x0 = (float*)xmalloc(sizeof(float) * N * N); 
  float *y0 = (float*)xmalloc(sizeof(float) * N * N); 
  float *z0 = (float*)xmalloc(sizeof(float) * N * N); 
  float *x1 = (float*)xmalloc(sizeof(float) * N * N); 
  float *y1 = (float*)xmalloc(sizeof(float) * N * N); 
  float *z1 = (float*)xmalloc(sizeof(float) * N * N); 

  start_timer();
  kernel(x0,y0,z0,x1,y1,z1);
  stop_timer();
  
  printf("%f\n", elapsed_time);



  if (atoi(argv[0]) == 9999999) {
    for (int i=0; i<N*N; i++) 
      printf("%f,%f,%f%f,%f,%f\n",x0[i],y0[i],z0[i],x1[i],y1[i],z1[i]);
  }
}
