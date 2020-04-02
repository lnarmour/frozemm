CC=icc
OPTS=-O3 -xcore-avx2
CFLAGS=-std=c99 -I/usr/include/malloc/ $(OPTS) -g
DEFS=

all: FMA

FMA.parametric: fma.c
	$(CC) fma.c -o FMA.parametric $(CFLAGS) $(DEFS) -qopt-report-phase=vec -qopt-report=5

FMA: fma.c
	$(CC) fma.c -o FMA $(CFLAGS) $(DEFS) -DCONSTANT=1 -qopt-report-phase=vec -qopt-report=5

clean:
	rm -f *.o FMA.parametric FMA.constant FMA
