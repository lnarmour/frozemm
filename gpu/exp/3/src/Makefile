OPTIONS := -O3 --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-6.4.0/bin  -std=c++11
EXECS := MM MM.check
all: $(EXECS) clean.o

debug: OPTIONS := -g -G --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-6.4.0/bin  -std=c++11
debug: all

clean:
	rm -f $(EXECS) *.o

timer.o : timer.cu timer.h
	nvcc $< -c -o $@ $(OPTIONS)

kernel.o : matmultKernel01.cu matmultKernel.h 
	nvcc $< -c -o $@ $(OPTIONS)

MM : matmult.cu  matmultKernel.h kernel.o timer.o
	nvcc $< kernel.o -o $@ $(LIB) timer.o $(OPTIONS)


MM.check : matmult.cu  matmultKernel.h kernel.o timer.o
	nvcc $< kernel.o -o $@ $(LIB) timer.o $(OPTIONS) -DCHECK=1 -Xcompiler -fopenmp

clean.o:
	rm -rf *.o

