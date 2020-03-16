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
#define mallocCheck(v,s,d) if ((v) == NULL) { printf("Failed to allocate memory for %s : size=%lu\n", "sizeof(d)*(s)", sizeof(d)*(s)); exit(-1); }

#define ALPHA 1.5
#define BETA 1.2

void MM_MKL(PRECISION, PRECISION, long, long, long, PRECISION*, PRECISION*, PRECISION*);
void MM(PRECISION, PRECISION, long, long, long, long, PRECISION*, PRECISION*, PRECISION*, double[3]);
void two2four(PRECISION*, PRECISION*, long, long, long, long);
void four2two(PRECISION*, PRECISION*, long, long, long, long);
void fetch_tile(PRECISION*, PRECISION*, long);
void update_tile(PRECISION*, long);

