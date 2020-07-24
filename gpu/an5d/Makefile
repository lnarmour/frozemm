ICUDA := -I /usr/local/cuda/include
INVML := -I /usr/local/cuda/nvml/include -lnvidia-ml -L /lib64 -L /usr/local/cuda/nvml/lib/
PWR := power/nvmlPower.cpp
IPWR := -I power

SM_TYPES := -gencode=arch=compute_61,code=sm_61

SB_TYPE := float

OPTIONS := -DSB_TYPE=$(SB_TYPE) -I ./src/ -I ./src/$(SB_TYPE)/ --use_fast_math $(SM_TYPES) -Xptxas -v -Xcompiler -fopenmp -O3

stencil := 
regs := 

all: $(stencil)

$(stencil): src/$(SB_TYPE)/$(stencil)_host.cu src/$(SB_TYPE)/$(stencil)_kernel.cu
	nvcc $(OPTIONS) $^ $(PWR) -o bin/$(SB_TYPE)/$@ $(ICUDA) $(INVML) $(IPWR) $(regs)

clean:
	rm -rf bin/float/* bin/double/*
