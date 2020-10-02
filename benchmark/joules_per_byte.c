#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


void kernel_jpb(double *X, long N) {

  for (long i=0; i<N; i++) {
    X[i] = i + 0.3;
  }

}


int main(int argc, char *argv[]) {

  if (argc < 2) {
    printf("usage %s G\n", argv[0]);
    return 0;
  }

  struct timeval t_start;
  struct timeval t_end;
  double etime;

  long gbs = atol(argv[1]);
  long size_array = gbs * (1<<30);
  long N = size_array / sizeof(double);

  double *X = malloc(size_array);

  gettimeofday (&t_start, NULL);

  kernel_jpb(X, N);

  gettimeofday (&t_end, NULL);

  etime = t_end.tv_sec - t_start.tv_sec +
        (t_end.tv_usec - t_start.tv_usec) * 1.0e-6;

  printf("execution time=%lf\n", etime);

  if (atoi(argv[0]) == 99999) {
    for (long i=0; i<N; i++)
      printf("%f-", X[i]);
  }


}
