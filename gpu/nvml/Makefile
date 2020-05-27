ICUDA := -I /usr/local/cuda/include 
INVML := -I /usr/local/cuda/nvml/include -lnvidia-ml -L /lib64 -L /usr/local/cuda/nvml/lib/
PWR := power/nvmlPower.cpp
IPWR := -I power
OPTIONS := -O3 --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-6.4.0/bin

all: transpose transpose_fma.1 transpose_fma.10 transpose_fma.100 mm

transpose: io-bound/transpose.cu $(PWR)
	nvcc $(ICUDA) -o bin/$@ $+ $(INVML) $(IPWR)

transpose_fma.1: compute-bound/transpose_fma.cu $(PWR)
	nvcc $(ICUDA) -o bin/$@ $+ $(INVML) $(IPWR) -DRUNS=1

transpose_fma.10: compute-bound/transpose_fma.cu $(PWR)
	nvcc $(ICUDA) -o bin/$@ $+ $(INVML) $(IPWR) -DRUNS=10

transpose_fma.100: compute-bound/transpose_fma.cu $(PWR)
	nvcc $(ICUDA) -o bin/$@ $+ $(INVML) $(IPWR) -DRUNS=100

timer.o: compute-bound/mm/timer.cu
	nvcc $< -c -o bin/$@ $(OPTIONS)

mm.o: compute-bound/mm/matmultKernel.cu
	nvcc $< -c -o bin/$@ $(OPTIONS)

mm: compute-bound/mm/matmult.cu mm.o timer.o $(PWR)
	nvcc $< bin/mm.o bin/timer.o $(PWR) -o bin/$@ $(LIB) $(OPTIONS) $(INVML) $(IPWR)


clean:
	rm -rf bin/*
