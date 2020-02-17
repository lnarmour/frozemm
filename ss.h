#ifdef INT
#define PRECISION long
#define CBLAS_GEMM cblas_sgemm
#endif

#ifdef SINGLE
#define PRECISION float
#define CBLAS_GEMM cblas_sgemm
#endif

#ifdef DOUBLE
#define PRECISION double
#define CBLAS_GEMM cblas_dgemm
#endif

#define start_timer(i) gettimeofday(&time, NULL); times[(i)] = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000)
#define stop_timer(i) gettimeofday(&time, NULL); times[(i)] = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000) - times[(i)]

#define min(x, y)   ((x)>(y) ? (y) : (x))
#define max(x, y)   ((x)>(y) ? (x) : (y))

void MM_MKL(long, long, long, PRECISION*, PRECISION*, PRECISION*);
void MM(long, long, long, long, PRECISION*, PRECISION*, PRECISION*, double[3]);
void two2four(PRECISION*, PRECISION*, long, long, long, long, long);
void four2two(PRECISION*, PRECISION*, long, long, long, long, long);
void two2four_single(PRECISION*, PRECISION*, long, long, long, long, long, long);
