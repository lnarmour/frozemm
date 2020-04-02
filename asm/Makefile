CC=icc
OPTS=-O3 -xcore-avx2
CFLAGS=-std=c99 -I/usr/include/malloc/ $(OPTS) -g
DEFS=

all: MM.simd MM

MM.simd: main.c
	$(CC) main.c -o MM.simd $(CFLAGS) $(DEFS) -DSIMD=1

MM: main.c
	$(CC) main.c -o MM $(CFLAGS) $(DEFS)

clean:
	rm -f *.o MM MM.simd
