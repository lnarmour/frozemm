#include "exp.h"

void kernel(long N, long B, double *X, long *order, long num_chunks)
{
  long o,ti,i;
  for (o=0; o<num_chunks; o++) {
    ti = order[o] * B;
    for (i=ti; i<min(N,ti+B); i++) {
      X[i] = 0.0;
//        printf("%4d ", i);
    }
//    printf("\n");
  }
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
//  for (i=0; i<num_chunks; i++)
//    printf("%ld -> %ld\n", i, order[i]);
}

void init_zero(double *Hog, long N)
{
  for (long i=0; i<N; i++)
    Hog[i] = 0.0;
}

int main(int argc, char** argv)
{
  if (argc <= 2) exit(1);
  long size_array = atol(argv[1]); // size (GBs) of array to allocate
  long size_chunks = atol(argv[2]);

  srand(time(0));

  long N = size_array / sizeof(double);
  long B = size_chunks / sizeof(double);
  double *X = xmalloc(N * sizeof(double));
  long num_chunks = (long)(N/B);
  long *order = xmalloc(num_chunks * sizeof(long));

  printf("N %ld, B %ld, num_chunks %ld\n", N, B, num_chunks);

  init_rand_chunk_order(order, num_chunks);  


  // init ~20 GB worth of memory a touch it first to push everything else to swap initially
  long _N = 2500000000;
       //_N = 2415919104; // 2^34 + 2^31
  double *Hog = xmalloc(_N * sizeof(double));
  init_zero(Hog, _N);
  if (size_array == 9999)
    printf("%f", Hog[size_array]);

  start_timer();
  kernel(N,B,X,order,num_chunks);
  stop_timer();

  printf("%f\n", elapsed_time);
}
