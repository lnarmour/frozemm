CC=icc
OPTS=-O3 -xcore-avx2
CFLAGS=-std=c99 -I/usr/include/malloc/ $(OPTS) -g
DEFS=

all: FMA.constant FMA.parametric

FMA.parametric: fma.c
	$(CC) fma.c -o FMA.parametric $(CFLAGS) $(DEFS) -qopt-report-phase=vec -qopt-report=5

FMA.constant: fma.c
	$(CC) fma.c -o FMA.constant $(CFLAGS) $(DEFS) -DN=2000 -DCONSTANT=1 

clean:
	rm -f *.o FMA.parametric FMA.constant FMA
