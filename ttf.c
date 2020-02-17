#include <stdio.h>
#include <stdlib.h>
#include "ss.h"

void two2four(double* I, double* O, long L, long M, long TSL, long TSM) {
	long tl, tm, l, m, i, j, u;
	for (i=0; i<L; i++)
		for (j=0; j<M; j++) {
			tl = i / TSL;
			tm = j / TSM;
			l = i % TSL;
			m = j % TSM;
			u = tl*(M*TSL) + tm*(TSL*TSM) + l*TSM + m;
			O[i*M + j] = I[u];
		}
}

void four2two(double* I, double* O, long L, long M, long TSL, long TSM) {
	long tl, tm, l, m, i, j, u;
	for (i=0; i<L; i++)
		for (j=0; j<M; j++) {
			tl = i / TSL;
			tm = j / TSM;
			l = i % TSL;
			m = j % TSM;
			u = tl*(M*TSL) + tm*(TSL*TSM) + l*TSM + m;
			O[u] = I[i*M + j];
		}
}

void printTile(PRECISION* X, long TS1, long TS2) {
  for (int i=0; i< TS1; i++) {
    for (int j=0; j<TS2; j++) {
      printf("%3.0f ", X[i*TS2+j]);
    }
    printf("\n");
  }
  printf("\n");
}


void setZero(double* X, long TS1, long TS2) {
	for (int i=0; i< TS1; i++) {
		for (int j=0; j<TS2; j++) {
			X[i*TS2+j] = 0;
		}
	}
}

int main() {

	long N = 8;
	long TSI = 4;
	long TSJ = 4;
	long TSK = 2;

	double* X1 = (double*) malloc(sizeof(double)*N*N);
	double* X2 = (double*) malloc(sizeof(double)*N*N);
	double* T24 = (double*) malloc(sizeof(double)*N*N);
	double* T42 = (double*) malloc(sizeof(double)*N*N);
	double* R_scratch = (double*) malloc(sizeof(double)*N*N);
	double* R = (double*) malloc(sizeof(double)*N*N);

	for (int i=0; i<N; i++) 
	for (int j=0; j<N; j++) {
		R_scratch[i*N+j] = 0;	
		X1[i*N+j] = 1; //i<4 && j<2 ? 1 : 0;
		X2[i*N+j] = 1; //i<2 && j<4 ? 1 : 0;
	}

	two2four(X1,T42,N,N,4,2);
	for (int i=0; i<N; i++) {
		for (int j=0; j<N; j++) {
			printf("%3.0f ", T42[i*N+j]);
		}
		printf("\n");
	}
	printf("\n");

	two2four(X2,T24,N,N,2,4);
	for (int i=0; i<N; i++) {
		for (int j=0; j<N; j++) {
			printf("%3.0f ", T24[i*N+j]);
		}
		printf("\n");
	}
	printf("\n");
	
	#define T42(tl,tm,TSL,TSM,L,M) T42[tl*(M*TSL) + tm*(TSL*TSM)]
	#define T24(tl,tm,TSL,TSM,L,M) T24[tl*(M*TSL) + tm*(TSL*TSM)]
	#define R_scratch(tl,tm,TSL,TSM,L,M) R_scratch[tl*(M*TSL) + tm*(TSL*TSM)]

	int ti, tj, tk;
	for (ti=0; ti<N/TSI; ti++)
		for (tj=0; tj<N/TSJ; tj++) 
			for (tk=0; tk<N/TSK; tk++) {
				printf("%d,%d,%d\n", ti, tj, tk);
				printTile(&T42(ti,tk,TSI,TSK,N,N), TSI, TSK);
				printTile(&T24(tk,tj,TSK,TSJ,N,N), TSK, TSJ);
				MM_MKL(TSI, TSK, TSJ, &T42(ti,tk,TSI,TSK,N,N), &T24(tk,tj,TSK,TSJ,N,N), &R_scratch(ti,tj,TSI,TSJ,N,N));
			}

	printTile(R_scratch, N, N);	
	four2two(R_scratch,R,N,N,TSI,TSJ);
	printTile(R, N, N);	

}
