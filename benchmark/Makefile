CC=icc
OPTS=-O3 -xcore-avx2
CFLAGS=-std=c99 -I/usr/include/malloc/ $(OPTS)
N=250

all: FMA

FMA: fma.c
	$(CC) fma.c -o FMA $(CFLAGS) -DN=$(N) -g

clean:
	rm -f *.o FMA
