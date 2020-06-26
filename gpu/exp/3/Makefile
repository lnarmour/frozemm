CUDA_VERSION=cuda
ICUDA := -I /usr/local/$(CUDA_VERSION)/include
INVML := -I /usr/local/$(CUDA_VERSION)/nvml/include -lnvidia-ml -L /lib64 -L /usr/local/$(CUDA_VERSION)/nvml/lib/
PWR := power/nvmlPower.cpp
IPWR := -I power
ARCH := sm_61

OPTIONS := -O3 --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-6.4.0/bin
EXECS := MM MM.check
all: $(EXECS)

debug: OPTIONS := -g -G --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-6.4.0/bin
debug: all

timer.o: src/timer.cu src/timer.h
	nvcc $< -c -o bin/$@ $(OPTIONS)

kernel.o: src/matmultKernel01.cu src/matmultKernel.h 
	nvcc $< -c -o bin/$@ $(OPTIONS)

MM: src/matmult.cu src/matmultKernel.h kernel.o timer.o
	nvcc src/matmult.cu $(PWR) bin/kernel.o bin/timer.o -o bin/$@ $(LIB) $(OPTIONS) $(ICUDA) $(INVML) $(IPWR)


MM.check: src/matmult.cu src/matmultKernel.h kernel.o timer.o
	nvcc src/matmult.cu $(PWR) bin/kernel.o bin/timer.o -o bin/$@ $(LIB) $(OPTIONS) -DCHECK=1 $(ICUDA) $(INVML) $(IPWR) -Xcompiler -fopenmp

clean:
	rm -rf bin/*

