ICUDA := -I /usr/local/cuda/include 
INVML := -I /usr/local/cuda/nvml/include -lnvidia-ml -L /lib64 -L /usr/local/cuda/nvml/lib/
PWR := power/nvmlPower.cpp
IPWR := -I power
OPTIONS :=  -O3 --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-6.4.0/bin 
RUNS := 1
EXECS := transpose transpose.nvprof
all: $(EXECS)

check.o: src/check.cu
	nvcc $+ -c -o bin/$@ $(OPTIONS) -Xcompiler -fopenmp

sgemm: src/sgemm.cu
	nvcc src/sgemm.cu $(PWR) -o bin/$@ $(OPTIONS) -lcublas -lgomp $(ICUDA) $(INVML) $(IPWR) -DRUNS=$(RUNS)

sgemv: src/sgemv.cu
	nvcc src/sgemv.cu $(PWR) -o bin/$@ $(OPTIONS) -lcublas -lgomp $(ICUDA) $(INVML) $(IPWR) -DRUNS=$(RUNS)

sgeam: src/sgeam.cu
	nvcc src/sgeam.cu $(PWR) -o bin/$@ $(OPTIONS) -lcublas -lgomp $(ICUDA) $(INVML) $(IPWR) -DRUNS=$(RUNS)

transpose: src/transpose.cu
	nvcc src/transpose.cu $(PWR) -o bin/$@ $(OPTIONS) $(ICUDA) $(INVML) $(IPWR)

transpose.nvprof: src/transpose.cu
	nvcc src/transpose.cu $(PWR) -o bin/$@ $(OPTIONS) $(ICUDA) $(INVML) $(IPWR) -DNVPROFILE=1



sgemm.check: src/sgemm.cu check.o
	nvcc src/sgemm.cu $(PWR) bin/check.o -o bin/$@ $(OPTIONS) -lcublas -lgomp -DCHECK $(ICUDA) $(INVML) $(IPWR)

sgemv.check: src/sgemv.cu check.o
	nvcc src/sgemv.cu $(PWR) bin/check.o -o bin/$@ $(OPTIONS) -lcublas -lgomp -DCHECK $(ICUDA) $(INVML) $(IPWR)

sgeam.check: src/sgeam.cu check.o
	nvcc src/sgeam.cu $(PWR) bin/check.o -o bin/$@ $(OPTIONS) -lcublas -lgomp -DCHECK $(ICUDA) $(INVML) $(IPWR)


clean:
	rm -rf bin/*
