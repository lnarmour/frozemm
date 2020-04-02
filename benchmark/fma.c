#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

struct timeval time;
double elapsed_time;
int posix_memalign(void **memptr, size_t alignment, size_t size);

#define min(x, y)   ((x)>(y) ? (y) : (x))
#define start_timer() gettimeofday(&time, NULL); elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000)
#define stop_timer() gettimeofday(&time, NULL); elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000) - elapsed_time

#ifdef CONSTANT
#ifndef N
#define N 2000
#endif
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

#ifdef CONSTANT
void kernel(float *restrict x0, float *restrict y0, float *restrict z0,
            float *restrict x1, float *restrict y1, float *restrict z1)
#else
void kernel(long N,
            float *restrict x0, float *restrict y0, float *restrict z0,
            float *restrict x1, float *restrict y1, float *restrict z1)
#endif
{
  int i,j,k;

  for (i=0; i<N; i+=1) {
    for (k=0; k<N; k+=1) {
      for (j=0; j<N; j+=1) {
        z0[i*N+j] += x0[i*N+k] * y0[k*N+j];
        z0[i*N+j] += x0[i*N+k] * y1[k*N+j];
      }
    }
  }



}

void main(int argc, char *argv[])
{
  #ifndef CONSTANT
  if (argc <=1) {
    printf("Usage: ./FMA.parametric N\n");
    return;
  }
  long N = atoi(argv[1]);
  #endif

  // x, y, & z should all fit in maxline L1d cache (32K bytes) if N < 1250
  float *x0 = (float*)xmalloc(sizeof(float) * N * N); 
  float *y0 = (float*)xmalloc(sizeof(float) * N * N); 
  float *z0 = (float*)xmalloc(sizeof(float) * N * N); 
  float *x1 = (float*)xmalloc(sizeof(float) * N * N); 
  float *y1 = (float*)xmalloc(sizeof(float) * N * N); 
  float *z1 = (float*)xmalloc(sizeof(float) * N * N); 

  start_timer();
  #ifdef CONSTANT
  kernel(x0,y0,z0,x1,y1,z1);
  #else
  kernel(N,x0,y0,z0,x1,y1,z1);
  #endif
  stop_timer();
  
  printf("%f\n", elapsed_time);



  if (atoi(argv[0]) == 9999999) {
    for (int i=0; i<N*N; i++) 
      printf("%f,%f,%f%f,%f,%f\n",x0[i],y0[i],z0[i],x1[i],y1[i],z1[i]);
  }
}
