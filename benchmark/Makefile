CC=icc
OPTS=-O3 -xcore-avx2
CFLAGS=-std=c99 -I/usr/include/malloc/ $(OPTS)
DEFS=

all: FMA

FMA: fma.scratch.c
	$(CC) fma.scratch.c -o FMA $(CFLAGS) $(DEFS)

clean:
	rm -f *.o FMA
