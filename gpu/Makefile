OPTIONS :=  -O3 --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-6.4.0/bin 
EXECS := MM MM.check
all: $(EXECS)
debug: OPTIONS := -g -G --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-4.6.3/bin/gcc
debug: all

timer.o: timer.cu timer.h
	nvcc $< -c -o $@ $(OPTIONS)

check.o: check.cu
	nvcc $< -c -o $@ $(OPTIONS)

MM: matmult.cu timer.o 
	nvcc $< -o $@ $(LIB) timer.o $(OPTIONS) -lcublas

MM.check: matmult.cu timer.o check.o
	nvcc $< -o $@ timer.o check.o $(OPTIONS) -lcublas -DCHECK

clean:
	rm -f $(EXECS) *.o
