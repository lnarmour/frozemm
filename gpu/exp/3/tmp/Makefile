OPTIONS   :=  -O3 --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-6.4.0/bin  -std=c++11
EXECS :=  matmult01
all:$(EXECS)

clean:
	rm -f $(EXECS) *.o

timer.o : timer.cu timer.h
	nvcc $< -c -o $@ $(OPTIONS)

matmultKernel01.o : matmultKernel01.cu matmultKernel.h 
	nvcc $< -c -o $@ $(OPTIONS) -DFOOTPRINT_SIZE_X=4 -DFOOTPRINT_SIZE_Y=4

matmult01 : matmult.cu  matmultKernel.h matmultKernel01.o timer.o
	nvcc $< matmultKernel01.o -o $@ $(LIB) timer.o $(OPTIONS) -DFOOTPRINT_SIZE_X=4 -DFOOTPRINT_SIZE_Y=4
