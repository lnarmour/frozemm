CC=icc
OPTS=-O3 -xcore-avx2
CFLAGS=-std=c99 -I/usr/include/malloc/ $(OPTS)
DEFS=

all: FMA.parametric FMA.constant

FMA.parametric: fma.c
	$(CC) fma.c -o FMA.parametric $(CFLAGS) $(DEFS)

FMA.constant: fma.c
	$(CC) fma.c -o FMA.constant $(CFLAGS) $(DEFS) -DCONSTANT=1

clean:
	rm -f *.o FMA.parametric FMA.constant
