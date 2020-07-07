ICUDA := -I /usr/local/$(CUDA_VERSION)/include 
INVML := -I /usr/local/$(CUDA_VERSION)/nvml/include -lnvidia-ml -L /lib64 -L /usr/local/$(CUDA_VERSION)/nvml/lib/
PWR := power/nvmlPower.cpp
IPWR := -I power
ARCH := sm_61
OPTIONS :=  -O3 --ptxas-options -v --gpu-architecture=$(ARCH) --compiler-bindir /usr/local/gcc-6.4.0/bin
RUNS := 1

EXECS := sgemm-ttss

all: $(EXECS)
debug: OPTIONS := -g -G --ptxas-options -v --gpu-architecture=$(ARCH) --compiler-bindir /usr/local/gcc-6.4.0/bin
debug: $(EXECS)

check.o: src/check.cu
	nvcc $+ -c -o bin/$@ $(OPTIONS) -Xcompiler -fopenmp

sgemm-ttss.check: src/matmult-ttss.cu check.o
	mkdir -p bin/$(HOSTNAME)
	nvcc src/matmult-ttss.cu $(PWR) bin/check.o -o bin/$(HOSTNAME)/$@ $(OPTIONS) -lcublas -lgomp -DCHECK $(ICUDA) $(INVML) $(IPWR) -DRUNS=$(RUNS)


sgemm-ttss: src/matmult-ttss.cu
	mkdir -p bin/$(HOSTNAME)
	nvcc src/matmult-ttss.cu $(PWR) -o bin/$(HOSTNAME)/$@.cuda.$(CV_NUM) $(OPTIONS) -lcublas -lgomp $(ICUDA) $(INVML) $(IPWR) -DRUNS=$(RUNS)

clean:
	rm -rf bin/$(HOSTNAME)
