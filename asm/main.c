#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <immintrin.h>

struct timeval time;
double elapsed_time;
int posix_memalign(void **memptr, size_t alignment, size_t size);

#define start_timer() gettimeofday(&time, NULL); elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000)
#define stop_timer() gettimeofday(&time, NULL); elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000) - elapsed_time

#ifndef N
#define N 32
#endif

#ifndef v
#define v 1.0
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

void kernel(float *x0, float *y0, float *z0, float *x1, float *y1, float *z1, long T)
{
  __m256 vx0, vy0, vz0, vr0;
  __m256 vx1, vy1, vz1, vr1;
  long t0,t1,k;
  for (t0=0; t0<T; t0++)
  for (t1=0; t1<T; t1++)
    #ifdef SIMD
    for (k=0; k<N; k+=8) {
      vx0 = _mm256_load_ps(&(x0[k]));
      vy0 = _mm256_load_ps(&(y0[k]));
      vz0 = _mm256_load_ps(&(z0[k]));
      vx1 = _mm256_load_ps(&(x1[k]));
      vy1 = _mm256_load_ps(&(y1[k]));
      vz1 = _mm256_load_ps(&(z1[k]));
      vr0 = _mm256_fmadd_ps(vx0, vy0, vz0);
      vr1 = _mm256_fmadd_ps(vx1, vy1, vz1);
      _mm256_store_ps(&(z0[k]), vr0);
      _mm256_store_ps(&(z1[k]), vr1);
    #else 
    for (k=0; k<N; k++) {
      z0[k]+=x0[k]*y0[k];
      z1[k]+=x1[k]*y1[k];
    #endif
    }
}

void main(int argc, char *argv[])
{
  if (argc <=1) return;
  int T = atoi(argv[1]);
  float *x0 = (float*)xmalloc(sizeof(float) * N);
  float *y0 = (float*)xmalloc(sizeof(float) * N);
  float *z0 = (float*)xmalloc(sizeof(float) * N);
  float *x1 = (float*)xmalloc(sizeof(float) * N);
  float *y1 = (float*)xmalloc(sizeof(float) * N);
  float *z1 = (float*)xmalloc(sizeof(float) * N);
  for (int i=0; i<N; i++) {
    x0[i] = v;
    y0[i] = v;
    z0[i] = 0.0;
    x1[i] = v;
    y1[i] = v;
    z1[i] = 0.0;
  }
  start_timer();
  kernel(x0,y0,z0,x1,y1,z1,T);
  stop_timer();
  
  printf("%f\n", elapsed_time);


  if (argc > 2)
  for (int i=0; i<N; i++) 
    printf("%2d  -  %.2f %.2f %.2f   |   %.2f %.2f %.2f\n",i,x0[i],y0[i],z0[i],x1[i],y1[i],z1[i]);
}
