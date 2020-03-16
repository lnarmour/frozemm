LIBRARIES=-lm
CC=icc
LD=xild
AR=xiar
MKL_FLAGS=-I${INTEL_HOME}/mkl/include/ -L${INTEL_HOME}/compilers_and_libraries/linux/mkl/lib/intel64  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core  -L${INTEL_HOME}/compilers_and_libraries/linux/lib/intel64 -liomp5 -lm
PRECISION=SINGLE
DEFS= 
OPTS= -O3 -xcore-avx2
CFLAGS=  -std=c99  -I/usr/include/malloc/ $(MKL_FLAGS) -qopenmp ${OPTS}
OBJS=ss-mkl.o ss.o
all: MM MM.check

debug: CFLAGS =-DDEBUG -O0 -g -Wall -Wextra -std=c99 -I/usr/include/malloc/ ${MKL_FLAGS}
debug: MM MM.check

MM: $(OBJS) ss-wrapper.c
	$(CC) ss-wrapper.c $(OBJS) -o MM $(CFLAGS) $(LIBRARIES) -D$(PRECISION)=1

MM.check: $(OBJS) ss-wrapper.c 
	$(CC) ss-wrapper.c $(OBJS) -o MM.check $(CFLAGS) $(LIBRARIES) -D$(PRECISION)=1 -DCHECK

ss-mkl.o: ss-mkl.c
	$(CC) ss-mkl.c -c -o ss-mkl.o $(CFLAGS) $(LIBRARIES) -D$(PRECISION)=1

ss.o: ss.c
	$(CC) ss.c -c -o ss.o $(CFLAGS) $(LIBRARIES) $(DEFS) -D$(PRECISION)=1 -qopt-report-phase=vec -qopt-report=1 -g

clean:
	rm -f *.o MM MM.*
