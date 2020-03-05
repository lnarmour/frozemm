#include "ss.h"

void fetch_tile(PRECISION* restrict X4d, PRECISION* restrict x, long NN) {
	#pragma omp parallel for
	for (long i=0; i<NN; i++)
		x[i] = X4d[i];
}

// ASSUMING that L%TSI=0 and M%TSJ=0
void two2four(PRECISION* restrict I, PRECISION* restrict scratch, long L, long M, long TSL, long TSM) {
  long ti, tl, tm, l, m, i, j, u;

    if (L == TSL && M == TSM)
        return;

	#pragma omp parallel for
	for (ti=0; ti<L/TSL; ti++) {
	  for (i=ti*TSL; i<(ti+1)*TSL; i++) {
	    for (j=0; j<M; j++) {
	      scratch[(i%TSL)*M + j] = I[i*M + j];
	    }
	  }
	  for (i=ti*TSL; i<(ti+1)*TSL; i++) {
        #pragma vector always
	    for (j=0; j<M; j++) {
	      tl = i / TSL;
	      tm = j / TSM;
	      l = i % TSL;
	      m = j % TSM;
	      u = tl*(M*TSL) + tm*(TSL*TSM) + l*TSM + m;
	      I[u] = scratch[(i%TSL)*M + j];
	    }
	  }
	}
}


// ASSUMING that L%TSI=0 and M%TSJ=0
void four2two(PRECISION* restrict I, PRECISION* restrict scratch, long L, long M, long TSL, long TSM) {
  long ti, tl, tm, l, m, i, j, u;

    if (L == TSL && M == TSM)
        return;

	#pragma omp parallel for
	for (ti=0; ti<L/TSL; ti++) {
	  for (i=ti*TSL; i<(ti+1)*TSL; i++) {
	    for (j=0; j<M; j++) {
	      scratch[(i%TSL)*M + j] = I[i*M + j];
	    }
	  }
	  for (i=ti*TSL; i<(ti+1)*TSL; i++) {
        #pragma vector always
	    for (j=0; j<M; j++) {
	      tm = j / TSM;
	      l = i % TSL;
	      m = j % TSM;
	      u = tm*(TSL*TSM) + l*TSM + m;
	      I[i*M + j] = scratch[u];
	    }
	  }
	}
}


