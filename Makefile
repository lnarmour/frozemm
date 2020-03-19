CC=icc
PRECISION=SINGLE
MKL_FLAGS=-I${INTEL_HOME}/mkl/include/ -L${INTEL_HOME}/compilers_and_libraries/linux/mkl/lib/intel64  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core  -L${INTEL_HOME}/compilers_and_libraries/linux/lib/intel64 -liomp5 -lm
DEFS= 
OPTS= -O3 -xcore-avx2
CFLAGS=  -std=c99  -I/usr/include/malloc/ ${OPTS}
OBJS=ss-mkl.o ss.o

all: MM

MM: ss.o ss-wrapper.c
	$(CC) ss-wrapper.c ss.o -o MM $(CFLAGS) -D$(PRECISION)=1 -g

MM.check: $(OBJS) ss-wrapper.c 
	$(CC) ss-wrapper.c $(OBJS) -o MM.check $(CFLAGS) $(MKL_FLAGS)  -D$(PRECISION)=1 -DCHECK -g

MM.mkl: ss-mkl.o ss-wrapper.c
	$(CC) ss-wrapper.c ss-mkl.o -o MM.mkl $(CFLAGS) $(MKL_FLAGS)  -D$(PRECISION)=1 -DMKL -g

ss-mkl.o: ss-mkl.c
	$(CC) ss-mkl.c -c -o ss-mkl.o $(CFLAGS) $(OPTS) $(MKL_FLAGS) -D$(PRECISION)=1 -g

ss.o: ss.c
	$(CC) ss.c -c -o ss.o $(CFLAGS) $(DEFS) -D$(PRECISION)=1 -qopt-report-phase=vec -qopt-report=5

clean:
	rm -f *.o MM MM.check
