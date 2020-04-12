CC=icc
MKL_FLAGS=-I${INTEL_HOME}/mkl/include/ -L${INTEL_HOME}/compilers_and_libraries/linux/mkl/lib/intel64  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core  -L${INTEL_HOME}/compilers_and_libraries/linux/lib/intel64 -liomp5
DEFS= 
OPTS=-O3 -xcore-avx2
RPT_FLAGS=-qopt-report-phase=vec -qopt-report=5
CFLAGS=-std=c99 -I/usr/include/malloc/ $(OPTS) -qopenmp

all: MM MM.check

MM: ss.c check.c
	$(CC) ss.c check.c -o MM $(CFLAGS) $(DEFS) $(RPT_FLAGS)

MM.check: ss.c check.c
	$(CC) ss.c check.c -o MM.check $(CFLAGS) $(DEFS) -DCHECK=1

clean:
	rm -f *.o MM MM.check
