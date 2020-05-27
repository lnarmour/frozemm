ICUDA := -I /usr/local/cuda/include 
INVML := -I /usr/local/cuda/nvml/include -lnvidia-ml -L /lib64 -L /usr/local/cuda/nvml/lib/
PWR := power/nvmlPower.cpp
IPWR := -I power

all: transpose transpose_fma

transpose: io-bound/transpose.cu $(PWR)
	nvcc $(ICUDA) -o bin/$@ $+ $(INVML) $(IPWR)

transpose_fma: compute-bound/transpose_fma.cu $(PWR)
	nvcc $(ICUDA) -o bin/$@ $+ $(INVML) $(IPWR)

clean:
	rm -rf bin/*
