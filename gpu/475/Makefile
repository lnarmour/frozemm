OPTIONS   :=  -O3 --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-6.4.0/bin 
EXECS := MM
all:$(EXECS)
debug:  OPTIONS := -g -G --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-4.6.3/bin/gcc
debug: all

clean:
	rm -f $(EXECS)  *.o
timer.o : timer.cu timer.h
	nvcc $< -c -o $@ $(OPTIONS)


mm.o : matmultKernel.cu matmultKernel.h 
	nvcc $< -c -o $@ $(OPTIONS)

MM : matmult.cu matmultKernel.h mm.o timer.o
	nvcc $< mm.o -o $@ $(LIB) timer.o $(OPTIONS)

