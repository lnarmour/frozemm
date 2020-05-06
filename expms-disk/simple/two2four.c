#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <time.h>

struct timeval ttime;
double elapsed_time;
int posix_memalign(void**, size_t, size_t);
int madvise(void*, size_t, int);

#define min(x, y) ((x)>(y) ? (y) : (x))
#define start_timer() gettimeofday(&ttime, NULL); elapsed_time = (((double) ttime.tv_sec) + ((double) ttime.tv_usec)/1000000)
#define stop_timer() gettimeofday(&ttime, NULL); elapsed_time = (((double) ttime.tv_sec) + ((double) ttime.tv_usec)/1000000) - elapsed_time

char ltime[19];

char* gltime()
{
  time_t now;
  time(&now);
  struct tm *local = localtime(&now); 
  int y,m,d,H,M,S;
  y = local->tm_year + 1900;
  m = local->tm_mon + 1;
  d = local->tm_mday;
  H = local->tm_hour;
  M = local->tm_min;
  S = local->tm_sec;
  //sprintf(ltime, "%02d-%02d-%02d %02d:%02d:%02d",y,m,d,H,M,S);
  sprintf(ltime, "%02d:%02d:%02d",H,M,S);
  return ltime;
}


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

void init_arr(double *X, long L, long M, long TS, char *lbl)
{
  long ti,i,j;
  for (ti=0; ti<L/TS; ti++) {
    printf("%s init %s ti %ld\n", gltime(), lbl, ti);
    for (i=ti*TS; i<min(L,(ti+1)*TS); i++) {
      for (j=0; j<M; j++) {
        X[i*M+j] = i*M+j;
      }
    }
  }
}

void two2four(double* restrict X, double* restrict scratch, long L, long M, long TSL, long TSM, char* lbl) {
  long ti, tl, tm, l, m, i, j, u;

  // struct rlimit *rlim;
  // getrlimit(RLIMIT_MEMLOCK, rlim);
  // printf("--> %d\n", rlim->rlim_cur);
  // printf("--> %d\n", rlim->rlim_max);

  for (ti=0; ti<L/TSL; ti++) {
    printf("%s two2four %s ti %ld (X->scratch)\n", gltime(), lbl, ti);
    for (i=ti*TSL; i<(ti+1)*TSL; i++) {
      for (j=0; j<M; j++) {
        scratch[(i%TSL)*M + j] = X[i*M + j];
      }
    }
    printf("%s two2four %s ti %ld (scratch->X)\n", gltime(), lbl, ti);
    for (i=ti*TSL; i<(ti+1)*TSL; i++) {
      for (j=0; j<M; j++) {
        tl = i / TSL;
        tm = j / TSM;
        l = i % TSL;
        m = j % TSM;
        u = tl*(M*TSL) + tm*(TSL*TSM) + l*TSM + m;
        X[u] = scratch[(i%TSL)*M + j];
      }
    }
  }

}



int main(int argc, char** argv)
{
  if (argc <=2) return -1;
  
  long N = atol(argv[1]);
  long TS = atol(argv[2]);

  double *A = xmalloc(N*N*sizeof(double));    // N=30k -> 7.2 GiB
  //double *B = xmalloc(N*N*sizeof(double));    // N=30k -> 7.2 GiB
  //double *C = xmalloc(N*N*sizeof(double));    // N=30k -> 7.2 GiB
  double *scratch = xmalloc(N*TS*sizeof(double));    // N=30k, TS=6K -> 1.44 GiB

  printf("A       -> %d GiB\n", (int)(N*N*sizeof(double)/1.0e9));
  //printf("B       -> %d GiB\n", (int)(N*N*sizeof(double)/1.0e9));
  //printf("C       -> %d GiB\n", (int)(N*N*sizeof(double)/1.0e9));
  printf("scratch -> %d GiB\n", (int)(N*TS*sizeof(double)/1.0e9));

  init_arr(A,N,N,TS, "A");
  //init_arr(B,N,N,TS, "B");
  //init_arr(C,N,N,TS, "C");

  start_timer();
  two2four(A, scratch, N, N, TS, TS, "A");
  //two2four(B, scratch, N, N, TS, TS, "B");
  //two2four(C, scratch, N, N, TS, TS, "C");
  stop_timer();

  printf("%f\n", elapsed_time);

  free(A);
//  free(B);
//  free(C);
  free(scratch);
}
