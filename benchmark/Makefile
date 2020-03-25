CC=icc
OPTS=-xcore-avx2
CFLAGS=-O3 -std=c99 -I/usr/include/malloc/ $(OPTS)
DEFS=

all: FMA.avx2 FMA.novec FMA.avx2.gcc FMA.novec.gcc

FMA: fma.c
	$(CC) fma.c -o FMA $(CFLAGS) $(DEFS)

scratch: fma.scratch.c
	$(CC) fma.scratch.c -o scratch $(CFLAGS) $(DEFS) -qopt-report-phase=vec -qopt-report=5

FMA.avx2: fma.c
	$(CC) fma.c -o FMA.avx2 -O3 -std=c99 -I/usr/include/malloc/ -xcore-avx2 -qopt-report-phase=vec -qopt-report=5 $(DEFS)

FMA.avx2.89: fma.89gflops.c
	$(CC) fma.89gflops.c -o FMA.avx2 -O3 -std=c99 -I/usr/include/malloc/ -xcore-avx2 -qopt-report-phase=vec -qopt-report=5 $(DEFS)

FMA.novec: fma.c
	$(CC) fma.c -o FMA.novec -O3 -std=c99 -I/usr/include/malloc/ -no-vec -qopt-report-phase=vec -qopt-report=5 $(DEFS)

FMA.avx2.gcc: fma.c
	gcc fma.c -o FMA.avx2.gcc -O3 -std=c99 -I/usr/include/malloc/ -O3 $(DEFS)

FMA.novec.gcc: fma.c
	$(CC) fma.c -o FMA.novec.gcc -O3 -std=c99 -I/usr/include/malloc/ -fno-tree-vectorize$(DEFS)

clean:
	rm -f *.o FMA FMA.avx2 FMA.novec FMA.*.gcc
