ICUDA := -I /usr/local/cuda/include
INVML := -I /usr/local/cuda/nvml/include -lnvidia-ml -L /lib64 -L /usr/local/cuda/nvml/lib/
PWR := power/nvmlPower.cpp
IPWR := -I power

OPTIONS := -I ./src/ --use_fast_math -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -Xptxas -v -Xcompiler -fopenmp -O3
bT := 1

all: star2d3r-512-$(bT)-512

star2d3r-512-$(bT)-512: src/star2d3r-512-$(bT)-512_kernel.cu src/star2d3r-512-$(bT)-512_host.cu
	nvcc $(OPTIONS) $^ $(PWR) -o bin/$@ $(ICUDA) $(INVML) $(IPWR)

clean:
	rm -rf bin/*
