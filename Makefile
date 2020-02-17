LIBRARIES=-lm
CC=icc
LD=xild
AR=xiar
MKL_FLAGS=-I${INTEL_HOME}/mkl/include/ -L${INTEL_HOME}/compilers_and_libraries/linux/mkl/lib/intel64  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core  -L${INTEL_HOME}/compilers_and_libraries/linux/lib/intel64 -liomp5 -lm
PRECISION=SINGLE
CFLAGS=  -std=c99  -I/usr/include/malloc/ $(MKL_FLAGS) -fopenmp -O3
all: MM MM.check

debug: CFLAGS =-DDEBUG -O0 -g -Wall -Wextra -std=c99 -I/usr/include/malloc/ ${MKL_FLAGS}
debug: MM MM.check
		
MM: ss.c ss-wrapper.c ss-mkl.c memory-layout.c
	$(CC) ss-wrapper.c ss.c ss-mkl.c memory-layout.c -o MM $(CFLAGS) $(LIBRARIES) -D$(PRECISION)=1

MM.check: ss.c ss-wrapper.c ss-mkl.c memory-layout.c
	$(CC) ss-wrapper.c ss.c ss-mkl.c memory-layout.c -o MM.check $(CFLAGS) $(LIBRARIES) -D$(PRECISION)=1 -DCHECK

ttf: ttf.c ss-mkl.c
	$(CC) ttf.c ss-mkl.c -o ttf $(CFLAGS) $(LIBRARIES) -DDOUBLE=1

clean:
	rm -f *.o MM MM.* ttf
