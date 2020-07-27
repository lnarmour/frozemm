ICUDA := -I /usr/local/cuda/include
INVML := -I /usr/local/cuda/nvml/include -lnvidia-ml -L /lib64 -L /usr/local/cuda/nvml/lib/
PWR := power/nvmlPower.cpp
IPWR := -I power
ROUND := 
SRC_DIR := src$(ROUND)
BIN_DIR := bin$(ROUND)
SM_TYPES := -gencode=arch=compute_61,code=sm_61

REGCOUNT :=  

SB_TYPE := float

OPTIONS := -DSB_TYPE=$(SB_TYPE) -I ./$(SRC_DIR)/ -I ./$(SRC_DIR)/$(SB_TYPE)/ --use_fast_math $(SM_TYPES) -Xptxas -v -Xcompiler -fopenmp -O3

stencil := 
regs := 

all: $(stencil)

$(stencil): $(SRC_DIR)/$(SB_TYPE)/$(stencil)_host.cu $(SRC_DIR)/$(SB_TYPE)/$(stencil)_kernel.cu
	mkdir -p $(BIN_DIR)/$(SB_TYPE)
	nvcc $(OPTIONS) $^ $(PWR) -o $(BIN_DIR)/$(SB_TYPE)/$@.r$(REGCOUNT) $(ICUDA) $(INVML) $(IPWR) $(regs)

clean:
	rm -rf $(BIN_DIR)/float/* $(BIN_DIR)/double/*
