#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl.h>

struct timeval time;
double elapsed_time;
int check(long, float*, float*, float*);
int posix_memalign(void**, size_t, size_t);

#define min(x, y) ((x)>(y) ? (y) : (x))
#define start_timer() gettimeofday(&time, NULL); elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000)
#define stop_timer() gettimeofday(&time, NULL); elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000) - elapsed_time


// took from PolyBench
extern void* xmalloc (size_t num)
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


void kernel(long N, long PI, long PJ, long TK, float *A, float *B, float *C, float *c)
{
  long pi,pj,tk,m,n,k,i,j;
  float *a, *b;

  if (N<=PI && N<=PJ) {
    c = C;
  }

  // outer two loops to iterate over (square) patches of C
  for (pi=0; pi<N; pi+=PI)
    for (pj=0; pj<N; pj+=PJ) {

      m = pi+PI<N ? PI : N-pi;
      n = pj+PJ<N ? PJ : N-pj;     
 
      for (i=0; i<m; i++) 
        for (j=0; j<n; j++) 
          c[i*n+j] = C[(pi+i)*N+(pj+j)];

      // for a given patch of C, make a series of MKL call with tall thin, short stout tiles of A & B
      for (tk=0; tk<N; tk+=TK) {
        a = &(A[pi*N+tk]);
        b = &(B[tk*N+pj]);
        k = tk+TK<N ? TK : N-tk;
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0,a,N,b,N,1.0,c,n);
      }
      
      for (i=0; i<m; i++) 
        for (j=0; j<n; j++)
          C[(pi+i)*N+(pj+j)] = c[i*n+j];
    }
}


int main(int argc, char** argv)
{
  if (argc <= 4) exit(1);
  long N = atoi(argv[1]);
  long PI = atoi(argv[2]);
  long PJ = atoi(argv[3]);
  long TK = atoi(argv[4]);

  float *A = xmalloc(N * N * sizeof(float));
  float *B = xmalloc(N * N * sizeof(float));
  float *C = xmalloc(N * N * sizeof(float));
  
  for (long i=0; i<N; i++)
    for (long j=0; j<N; j++) {
      // took from PolyBench
      C[i*N+j] = (float) ((i*j+1) % N) / N;
      A[i*N+j] = (float) (i*(j+1) % N) / N;
      B[i*N+j] = (float) (i*(j+2) % N) / N;
   }

  float *c = xmalloc(PI * PJ * sizeof(float));

  start_timer();
  kernel(N,PI,PJ,TK,A,B,C,c);
  stop_timer();

  printf("%f\n", elapsed_time);

  #ifdef CHECK
  check(N,A,B,C);
  #endif

}
