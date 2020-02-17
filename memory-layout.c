#include "ss.h"


// ASSUMING that L%TSI=0 and M%TSJ=0
void two2four(PRECISION* I, PRECISION* scratch, long L, long M, long TSL, long TSM, long ti) {
  long tl, tm, l, m, i, j, u;
  for (i=ti*TSL; i<(ti+1)*TSL; i++) {
    for (j=0; j<M; j++) {
      scratch[(i%TSL)*M + j] = I[i*M + j];
    }
  }
  for (i=ti*TSL; i<(ti+1)*TSL; i++) {
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


// ASSUMING that L%TSI=0 and M%TSJ=0
void four2two(PRECISION* I, PRECISION* scratch, long L, long M, long TSL, long TSM, long ti) {
  long tl, tm, l, m, i, j, u;
  for (i=ti*TSL; i<(ti+1)*TSL; i++) {
    for (j=0; j<M; j++) {
      scratch[(i%TSL)*M + j] = I[i*M + j];
    }
  }
  for (i=ti*TSL; i<(ti+1)*TSL; i++) {
    for (j=0; j<M; j++) {
      tl = i / TSL;
      tm = j / TSM;
      l = i % TSL;
      m = j % TSM;
      u = tm*(TSL*TSM) + l*TSM + m;
      I[i*M + j] = scratch[u];
    }
  }
}

// -------------------------------

void two2four_single(PRECISION* I, PRECISION* scratch, long L, long M, long TSL, long TSM, long tl, long tm) {
	// put the elements of tile at tl_start and tm_start in 4D layout in scratch
	long l, m, i, j, u;
	long i_start = tl*TSL;
	long j_start = tm*TSM;

	for (i=i_start; i<i_start+TSL; i++)
		for (j=j_start; j<j_start+TSM; j++) {
			u = tm*(TSL*TSM) + (i-i_start)*TSM + (j-j_start);
			scratch[u] = I[i*M + j];
		}
}
































