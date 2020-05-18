#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

struct timeval time;
double elapsed_time;
int posix_memalign(void **memptr, size_t alignment, size_t size);

#define start_timer() gettimeofday(&time, NULL); elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000)
#define stop_timer() gettimeofday(&time, NULL); elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000) - elapsed_time

//#ifndef N
//#define N 2000
//#endif

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

void kernel(long N,
            float *restrict x0, float *restrict y0, float *restrict z0,
            float *restrict x1, float *restrict y1, float *restrict z1)
{
  int i,j,k;
  #ifdef version0
  // shared_writes(False) shared_reads(False) unroll(1,1,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version1
  // shared_writes(False) shared_reads(False) unroll(1,1,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version2
  // shared_writes(False) shared_reads(False) unroll(1,1,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version3
  // shared_writes(False) shared_reads(False) unroll(1,2,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version4
  // shared_writes(False) shared_reads(False) unroll(1,2,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version5
  // shared_writes(False) shared_reads(False) unroll(1,2,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version6
  // shared_writes(False) shared_reads(False) unroll(1,4,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version7
  // shared_writes(False) shared_reads(False) unroll(1,4,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version8
  // shared_writes(False) shared_reads(False) unroll(1,4,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version9
  // shared_writes(False) shared_reads(False) unroll(2,1,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version10
  // shared_writes(False) shared_reads(False) unroll(2,1,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version11
  // shared_writes(False) shared_reads(False) unroll(2,1,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version12
  // shared_writes(False) shared_reads(False) unroll(2,2,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version13
  // shared_writes(False) shared_reads(False) unroll(2,2,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version14
  // shared_writes(False) shared_reads(False) unroll(2,2,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version15
  // shared_writes(False) shared_reads(False) unroll(2,4,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version16
  // shared_writes(False) shared_reads(False) unroll(2,4,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version17
  // shared_writes(False) shared_reads(False) unroll(2,4,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version18
  // shared_writes(False) shared_reads(False) unroll(4,1,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version19
  // shared_writes(False) shared_reads(False) unroll(4,1,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version20
  // shared_writes(False) shared_reads(False) unroll(4,1,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version21
  // shared_writes(False) shared_reads(False) unroll(4,2,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version22
  // shared_writes(False) shared_reads(False) unroll(4,2,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version23
  // shared_writes(False) shared_reads(False) unroll(4,2,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version24
  // shared_writes(False) shared_reads(False) unroll(4,4,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version25
  // shared_writes(False) shared_reads(False) unroll(4,4,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version26
  // shared_writes(False) shared_reads(False) unroll(4,4,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+3)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+3)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version27
  // shared_writes(False) shared_reads(False) unroll(1,1,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version28
  // shared_writes(False) shared_reads(False) unroll(1,1,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version29
  // shared_writes(False) shared_reads(False) unroll(1,1,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version30
  // shared_writes(False) shared_reads(False) unroll(1,2,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version31
  // shared_writes(False) shared_reads(False) unroll(1,2,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version32
  // shared_writes(False) shared_reads(False) unroll(1,2,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version33
  // shared_writes(False) shared_reads(False) unroll(1,4,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version34
  // shared_writes(False) shared_reads(False) unroll(1,4,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version35
  // shared_writes(False) shared_reads(False) unroll(1,4,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version36
  // shared_writes(False) shared_reads(False) unroll(2,1,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version37
  // shared_writes(False) shared_reads(False) unroll(2,1,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version38
  // shared_writes(False) shared_reads(False) unroll(2,1,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version39
  // shared_writes(False) shared_reads(False) unroll(2,2,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version40
  // shared_writes(False) shared_reads(False) unroll(2,2,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version41
  // shared_writes(False) shared_reads(False) unroll(2,2,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version42
  // shared_writes(False) shared_reads(False) unroll(2,4,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version43
  // shared_writes(False) shared_reads(False) unroll(2,4,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version44
  // shared_writes(False) shared_reads(False) unroll(2,4,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version45
  // shared_writes(False) shared_reads(False) unroll(4,1,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version46
  // shared_writes(False) shared_reads(False) unroll(4,1,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version47
  // shared_writes(False) shared_reads(False) unroll(4,1,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version48
  // shared_writes(False) shared_reads(False) unroll(4,2,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version49
  // shared_writes(False) shared_reads(False) unroll(4,2,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version50
  // shared_writes(False) shared_reads(False) unroll(4,2,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version51
  // shared_writes(False) shared_reads(False) unroll(4,4,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version52
  // shared_writes(False) shared_reads(False) unroll(4,4,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version53
  // shared_writes(False) shared_reads(False) unroll(4,4,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version54
  // shared_writes(False) shared_reads(True) unroll(1,1,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version55
  // shared_writes(False) shared_reads(True) unroll(1,1,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version56
  // shared_writes(False) shared_reads(True) unroll(1,1,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version57
  // shared_writes(False) shared_reads(True) unroll(1,2,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version58
  // shared_writes(False) shared_reads(True) unroll(1,2,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version59
  // shared_writes(False) shared_reads(True) unroll(1,2,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version60
  // shared_writes(False) shared_reads(True) unroll(1,4,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version61
  // shared_writes(False) shared_reads(True) unroll(1,4,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version62
  // shared_writes(False) shared_reads(True) unroll(1,4,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version63
  // shared_writes(False) shared_reads(True) unroll(2,1,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version64
  // shared_writes(False) shared_reads(True) unroll(2,1,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version65
  // shared_writes(False) shared_reads(True) unroll(2,1,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version66
  // shared_writes(False) shared_reads(True) unroll(2,2,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version67
  // shared_writes(False) shared_reads(True) unroll(2,2,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version68
  // shared_writes(False) shared_reads(True) unroll(2,2,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version69
  // shared_writes(False) shared_reads(True) unroll(2,4,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version70
  // shared_writes(False) shared_reads(True) unroll(2,4,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version71
  // shared_writes(False) shared_reads(True) unroll(2,4,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version72
  // shared_writes(False) shared_reads(True) unroll(4,1,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version73
  // shared_writes(False) shared_reads(True) unroll(4,1,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version74
  // shared_writes(False) shared_reads(True) unroll(4,1,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version75
  // shared_writes(False) shared_reads(True) unroll(4,2,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version76
  // shared_writes(False) shared_reads(True) unroll(4,2,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version77
  // shared_writes(False) shared_reads(True) unroll(4,2,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version78
  // shared_writes(False) shared_reads(True) unroll(4,4,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version79
  // shared_writes(False) shared_reads(True) unroll(4,4,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version80
  // shared_writes(False) shared_reads(True) unroll(4,4,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+3)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+3)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version81
  // shared_writes(False) shared_reads(True) unroll(1,1,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version82
  // shared_writes(False) shared_reads(True) unroll(1,1,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version83
  // shared_writes(False) shared_reads(True) unroll(1,1,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version84
  // shared_writes(False) shared_reads(True) unroll(1,2,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version85
  // shared_writes(False) shared_reads(True) unroll(1,2,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version86
  // shared_writes(False) shared_reads(True) unroll(1,2,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version87
  // shared_writes(False) shared_reads(True) unroll(1,4,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version88
  // shared_writes(False) shared_reads(True) unroll(1,4,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version89
  // shared_writes(False) shared_reads(True) unroll(1,4,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version90
  // shared_writes(False) shared_reads(True) unroll(2,1,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version91
  // shared_writes(False) shared_reads(True) unroll(2,1,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version92
  // shared_writes(False) shared_reads(True) unroll(2,1,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version93
  // shared_writes(False) shared_reads(True) unroll(2,2,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version94
  // shared_writes(False) shared_reads(True) unroll(2,2,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version95
  // shared_writes(False) shared_reads(True) unroll(2,2,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version96
  // shared_writes(False) shared_reads(True) unroll(2,4,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version97
  // shared_writes(False) shared_reads(True) unroll(2,4,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version98
  // shared_writes(False) shared_reads(True) unroll(2,4,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version99
  // shared_writes(False) shared_reads(True) unroll(4,1,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version100
  // shared_writes(False) shared_reads(True) unroll(4,1,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version101
  // shared_writes(False) shared_reads(True) unroll(4,1,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version102
  // shared_writes(False) shared_reads(True) unroll(4,2,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version103
  // shared_writes(False) shared_reads(True) unroll(4,2,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version104
  // shared_writes(False) shared_reads(True) unroll(4,2,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version105
  // shared_writes(False) shared_reads(True) unroll(4,4,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version106
  // shared_writes(False) shared_reads(True) unroll(4,4,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version107
  // shared_writes(False) shared_reads(True) unroll(4,4,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version108
  // shared_writes(True) shared_reads(False) unroll(1,1,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version109
  // shared_writes(True) shared_reads(False) unroll(1,1,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version110
  // shared_writes(True) shared_reads(False) unroll(1,1,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version111
  // shared_writes(True) shared_reads(False) unroll(1,2,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version112
  // shared_writes(True) shared_reads(False) unroll(1,2,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version113
  // shared_writes(True) shared_reads(False) unroll(1,2,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version114
  // shared_writes(True) shared_reads(False) unroll(1,4,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version115
  // shared_writes(True) shared_reads(False) unroll(1,4,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version116
  // shared_writes(True) shared_reads(False) unroll(1,4,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version117
  // shared_writes(True) shared_reads(False) unroll(2,1,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version118
  // shared_writes(True) shared_reads(False) unroll(2,1,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version119
  // shared_writes(True) shared_reads(False) unroll(2,1,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version120
  // shared_writes(True) shared_reads(False) unroll(2,2,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version121
  // shared_writes(True) shared_reads(False) unroll(2,2,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version122
  // shared_writes(True) shared_reads(False) unroll(2,2,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version123
  // shared_writes(True) shared_reads(False) unroll(2,4,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version124
  // shared_writes(True) shared_reads(False) unroll(2,4,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version125
  // shared_writes(True) shared_reads(False) unroll(2,4,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version126
  // shared_writes(True) shared_reads(False) unroll(4,1,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version127
  // shared_writes(True) shared_reads(False) unroll(4,1,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version128
  // shared_writes(True) shared_reads(False) unroll(4,1,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version129
  // shared_writes(True) shared_reads(False) unroll(4,2,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version130
  // shared_writes(True) shared_reads(False) unroll(4,2,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version131
  // shared_writes(True) shared_reads(False) unroll(4,2,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version132
  // shared_writes(True) shared_reads(False) unroll(4,4,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version133
  // shared_writes(True) shared_reads(False) unroll(4,4,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version134
  // shared_writes(True) shared_reads(False) unroll(4,4,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+3)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+3)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version135
  // shared_writes(True) shared_reads(False) unroll(1,1,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version136
  // shared_writes(True) shared_reads(False) unroll(1,1,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version137
  // shared_writes(True) shared_reads(False) unroll(1,1,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version138
  // shared_writes(True) shared_reads(False) unroll(1,2,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version139
  // shared_writes(True) shared_reads(False) unroll(1,2,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version140
  // shared_writes(True) shared_reads(False) unroll(1,2,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version141
  // shared_writes(True) shared_reads(False) unroll(1,4,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version142
  // shared_writes(True) shared_reads(False) unroll(1,4,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version143
  // shared_writes(True) shared_reads(False) unroll(1,4,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version144
  // shared_writes(True) shared_reads(False) unroll(2,1,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version145
  // shared_writes(True) shared_reads(False) unroll(2,1,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version146
  // shared_writes(True) shared_reads(False) unroll(2,1,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version147
  // shared_writes(True) shared_reads(False) unroll(2,2,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version148
  // shared_writes(True) shared_reads(False) unroll(2,2,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version149
  // shared_writes(True) shared_reads(False) unroll(2,2,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version150
  // shared_writes(True) shared_reads(False) unroll(2,4,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version151
  // shared_writes(True) shared_reads(False) unroll(2,4,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version152
  // shared_writes(True) shared_reads(False) unroll(2,4,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version153
  // shared_writes(True) shared_reads(False) unroll(4,1,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version154
  // shared_writes(True) shared_reads(False) unroll(4,1,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version155
  // shared_writes(True) shared_reads(False) unroll(4,1,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version156
  // shared_writes(True) shared_reads(False) unroll(4,2,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version157
  // shared_writes(True) shared_reads(False) unroll(4,2,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version158
  // shared_writes(True) shared_reads(False) unroll(4,2,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version159
  // shared_writes(True) shared_reads(False) unroll(4,4,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version160
  // shared_writes(True) shared_reads(False) unroll(4,4,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version161
  // shared_writes(True) shared_reads(False) unroll(4,4,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version162
  // shared_writes(True) shared_reads(True) unroll(1,1,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version163
  // shared_writes(True) shared_reads(True) unroll(1,1,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version164
  // shared_writes(True) shared_reads(True) unroll(1,1,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version165
  // shared_writes(True) shared_reads(True) unroll(1,2,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version166
  // shared_writes(True) shared_reads(True) unroll(1,2,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version167
  // shared_writes(True) shared_reads(True) unroll(1,2,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version168
  // shared_writes(True) shared_reads(True) unroll(1,4,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version169
  // shared_writes(True) shared_reads(True) unroll(1,4,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version170
  // shared_writes(True) shared_reads(True) unroll(1,4,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version171
  // shared_writes(True) shared_reads(True) unroll(2,1,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version172
  // shared_writes(True) shared_reads(True) unroll(2,1,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version173
  // shared_writes(True) shared_reads(True) unroll(2,1,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version174
  // shared_writes(True) shared_reads(True) unroll(2,2,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version175
  // shared_writes(True) shared_reads(True) unroll(2,2,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version176
  // shared_writes(True) shared_reads(True) unroll(2,2,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version177
  // shared_writes(True) shared_reads(True) unroll(2,4,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version178
  // shared_writes(True) shared_reads(True) unroll(2,4,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version179
  // shared_writes(True) shared_reads(True) unroll(2,4,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version180
  // shared_writes(True) shared_reads(True) unroll(4,1,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version181
  // shared_writes(True) shared_reads(True) unroll(4,1,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version182
  // shared_writes(True) shared_reads(True) unroll(4,1,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+0)];
      }
    }
  }
  #endif

  #ifdef version183
  // shared_writes(True) shared_reads(True) unroll(4,2,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version184
  // shared_writes(True) shared_reads(True) unroll(4,2,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version185
  // shared_writes(True) shared_reads(True) unroll(4,2,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+1)];
      }
    }
  }
  #endif

  #ifdef version186
  // shared_writes(True) shared_reads(True) unroll(4,4,1) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version187
  // shared_writes(True) shared_reads(True) unroll(4,4,2) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version188
  // shared_writes(True) shared_reads(True) unroll(4,4,4) perm(i.j.i.k.k.j)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+0)*N+(j+0)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+0)*N+(j+1)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+0)*N+(j+2)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+0)*N+(j+3)] += x0[(i+0)*N+(k+3)] * y0[(k+3)*N+(j+3)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+1)*N+(j+0)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+1)*N+(j+1)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+1)*N+(j+2)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+1)*N+(j+3)] += x0[(i+1)*N+(k+3)] * y0[(k+3)*N+(j+3)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+2)*N+(j+0)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+2)*N+(j+1)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+2)*N+(j+2)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+2)*N+(j+3)] += x0[(i+2)*N+(k+3)] * y0[(k+3)*N+(j+3)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+0)];
        z0[(i+3)*N+(j+0)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+0)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+1)];
        z0[(i+3)*N+(j+1)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+1)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+2)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+2)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+2)];
        z0[(i+3)*N+(j+2)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+2)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+0)] * y0[(k+0)*N+(j+3)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+1)] * y0[(k+1)*N+(j+3)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+2)] * y0[(k+2)*N+(j+3)];
        z0[(i+3)*N+(j+3)] += x0[(i+3)*N+(k+3)] * y0[(k+3)*N+(j+3)];
      }
    }
  }
  #endif

  #ifdef version189
  // shared_writes(True) shared_reads(True) unroll(1,1,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version190
  // shared_writes(True) shared_reads(True) unroll(1,1,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version191
  // shared_writes(True) shared_reads(True) unroll(1,1,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version192
  // shared_writes(True) shared_reads(True) unroll(1,2,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version193
  // shared_writes(True) shared_reads(True) unroll(1,2,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version194
  // shared_writes(True) shared_reads(True) unroll(1,2,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version195
  // shared_writes(True) shared_reads(True) unroll(1,4,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version196
  // shared_writes(True) shared_reads(True) unroll(1,4,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version197
  // shared_writes(True) shared_reads(True) unroll(1,4,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=1) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version198
  // shared_writes(True) shared_reads(True) unroll(2,1,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version199
  // shared_writes(True) shared_reads(True) unroll(2,1,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version200
  // shared_writes(True) shared_reads(True) unroll(2,1,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version201
  // shared_writes(True) shared_reads(True) unroll(2,2,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version202
  // shared_writes(True) shared_reads(True) unroll(2,2,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version203
  // shared_writes(True) shared_reads(True) unroll(2,2,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version204
  // shared_writes(True) shared_reads(True) unroll(2,4,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version205
  // shared_writes(True) shared_reads(True) unroll(2,4,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version206
  // shared_writes(True) shared_reads(True) unroll(2,4,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=2) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version207
  // shared_writes(True) shared_reads(True) unroll(4,1,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version208
  // shared_writes(True) shared_reads(True) unroll(4,1,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version209
  // shared_writes(True) shared_reads(True) unroll(4,1,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=1) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version210
  // shared_writes(True) shared_reads(True) unroll(4,2,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version211
  // shared_writes(True) shared_reads(True) unroll(4,2,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version212
  // shared_writes(True) shared_reads(True) unroll(4,2,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=2) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+3)];
      }
    }
  }
  #endif

  #ifdef version213
  // shared_writes(True) shared_reads(True) unroll(4,4,1) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=1) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+0)];
      }
    }
  }
  #endif

  #ifdef version214
  // shared_writes(True) shared_reads(True) unroll(4,4,2) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=2) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+1)];
      }
    }
  }
  #endif

  #ifdef version215
  // shared_writes(True) shared_reads(True) unroll(4,4,4) perm(i.k.i.j.j.k)
  for (i=0; i<N; i+=4) {
    for (j=0; j<N; j+=4) {
      for (k=0; k<N; k+=4) {
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+0)*N+(k+0)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+0)*N+(k+1)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+0)*N+(k+2)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+0)*N+(k+3)] += x0[(i+0)*N+(j+3)] * y0[(j+3)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+1)*N+(k+0)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+1)*N+(k+1)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+1)*N+(k+2)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+1)*N+(k+3)] += x0[(i+1)*N+(j+3)] * y0[(j+3)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+2)*N+(k+0)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+2)*N+(k+1)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+2)*N+(k+2)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+2)*N+(k+3)] += x0[(i+2)*N+(j+3)] * y0[(j+3)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+0)] * y0[(j+0)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+1)] * y0[(j+1)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+2)] * y0[(j+2)*N+(k+3)];
        z0[(i+3)*N+(k+0)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+0)];
        z0[(i+3)*N+(k+1)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+1)];
        z0[(i+3)*N+(k+2)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+2)];
        z0[(i+3)*N+(k+3)] += x0[(i+3)*N+(j+3)] * y0[(j+3)*N+(k+3)];
      }
    }
  }
  #endif

}


void main(int argc, char *argv[])
{
  if (argc <= 1) exit(1);
  long N = atoi(argv[1]);

  // x, y, & z should all fit in maxline L1d cache (32K bytes) if N < 1250
  float *x0 = (float*)xmalloc(sizeof(float) * N * N); 
  float *y0 = (float*)xmalloc(sizeof(float) * N * N); 
  float *z0 = (float*)xmalloc(sizeof(float) * N * N); 
  float *x1 = (float*)xmalloc(sizeof(float) * N * N); 
  float *y1 = (float*)xmalloc(sizeof(float) * N * N); 
  float *z1 = (float*)xmalloc(sizeof(float) * N * N); 

  start_timer();
  kernel(N, x0,y0,z0,x1,y1,z1);
  stop_timer();
  
  printf("%f\n", elapsed_time);



  if (atoi(argv[0]) == 9999999) {
    for (int i=0; i<N*N; i++) 
      printf("%f,%f,%f%f,%f,%f\n",x0[i],y0[i],z0[i],x1[i],y1[i],z1[i]);
  }
}