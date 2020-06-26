OPTIONS := -O3 --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-6.4.0/bin  -std=c++11
EXECS := matmult01
all:$(EXECS)

debug: OPTIONS := -g -G --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-6.4.0/bin  -std=c++11
debug: all

clean:
	rm -f $(EXECS) *.o

timer.o : timer.cu timer.h
	nvcc $< -c -o $@ $(OPTIONS)

matmultKernel01.o : matmultKernel01.cu matmultKernel.h 
	nvcc $< -c -o $@ $(OPTIONS)

matmult01 : matmult.cu  matmultKernel.h matmultKernel01.o timer.o
	nvcc $< matmultKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)
