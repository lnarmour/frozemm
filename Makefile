LIBRARIES=-lm
CC=icc
LD=xild
AR=xiar
MKL_FLAGS=-I${INTEL_HOME}/mkl/include/ -L${INTEL_HOME}/compilers_and_libraries/linux/mkl/lib/intel64  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core  -L${INTEL_HOME}/compilers_and_libraries/linux/lib/intel64 -liomp5 -lm
PRECISION=SINGLE
CFLAGS=  -std=c99  -I/usr/include/malloc/ $(MKL_FLAGS) -fopenmp -O3
OBJS=ss-mkl.o memory-layout.o
all: MM.sequential MM.parallel MM.check wtf.sequential wtf.parallel

debug: CFLAGS =-DDEBUG -O0 -g -Wall -Wextra -std=c99 -I/usr/include/malloc/ ${MKL_FLAGS}
debug: MM.sequential MM.parallel MM.check
		
MM.sequential: ss_seq.o $(OBJS) ss-wrapper.c
	$(CC) ss-wrapper.c ss_seq.o $(OBJS) -o MM.sequential $(CFLAGS) $(LIBRARIES) -D$(PRECISION)=1

MM.parallel: ss_par.o $(OBJS) ss-wrapper.c
	$(CC) ss-wrapper.c ss_par.o $(OBJS) -o MM.parallel $(CFLAGS) $(LIBRARIES) -D$(PRECISION)=1

MM.check: ss_seq.o $(OBJS) ss-wrapper.c 
	$(CC) ss-wrapper.c ss_seq.o $(OBJS) -o MM.check $(CFLAGS) $(LIBRARIES) -D$(PRECISION)=1 -DCHECK

ss-mkl.o: ss-mkl.c
	$(CC) ss-mkl.c -c -o ss-mkl.o $(CFLAGS) $(LIBRARIES) -D$(PRECISION)=1

ss_seq.o: ss.c
	$(CC) ss.c -c -o ss_seq.o $(CFLAGS) $(LIBRARIES) -D$(PRECISION)=1

ss_par.o: ss.c
	$(CC) ss.c -c -o ss_par.o $(CFLAGS) $(LIBRARIES) -D$(PRECISION)=1 -DPARALLEL=1

memory-layout.o: memory-layout.c
	$(CC) memory-layout.c -c -o memory-layout.o $(CFLAGS) $(LIBRARIES) -D$(PRECISION)=1

wtf.sequential: wtf.c memory-layout.o ss-mkl.o
	$(CC) wtf.c memory-layout.o ss-mkl.o -o wtf.sequential $(CFLAGS) $(LIBRARIES) -D$(PRECISION)=1 -DSEQUENTIAL=1

wtf.parallel: wtf.c memory-layout.o ss-mkl.o
	$(CC) wtf.c memory-layout.o ss-mkl.o -o wtf.parallel $(CFLAGS) $(LIBRARIES) -D$(PRECISION)=1

clean:
	rm -f *.o MM.* wtf.sequential wtf.parallel
