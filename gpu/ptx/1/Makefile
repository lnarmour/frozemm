ARCH=sm_30

all: example


example.cubin: example.ptx
	ptxas -arch=$(ARCH) -m64 $< -o $@

example.fatbin: example.cubin example.ptx
	fatbinary --create=$@ -64 "--image=profile=$(ARCH),file=$(word 1,$^)" "--image=profile=compute_30,file=$(word 2,$^)" --embedded-fatbin="$@.c"

example.cpp4.ii: example.cu
	gcc -E -x c++ -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include" -D__CUDACC_VER_MAJOR__=10 -D__CUDACC_VER_MINOR__=1 -D__CUDACC_VER_BUILD__=243 -include "cuda_runtime.h" -m64 $< > $@ 

example.cudafe1.cpp: example.cpp4.ii
	cudafe++ --c++14 --gnu_version=70400 --allow_managed  --m64 --parse_templates --gen_c_file_name "/tmp/tmpxft_0025a581_00000000-5_example.cudafe1.cpp" --stub_file_name "tmpxft_0025a581_00000000-5_example.cudafe1.stub.c" --module_id_file_name "/tmp/tmpxft_0025a581_00000000-3_example.module_id" "example.cpp4.ii" 

example: example.cu example.cudafe1.cpp 
	gcc -D__CUDA_ARCH__=300 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -m64 -o "/tmp/tmpxft_0025a581_00000000-10_example.o" "/tmp/tmpxft_0025a581_00000000-5_example.cudafe1.cpp"
	nvlink --arch=sm_30 --register-link-binaries="/tmp/tmpxft_0025a581_00000000-6_example_dlink.reg.c"  -m64   "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -cpu-arch=X86_64 "/tmp/tmpxft_0025a581_00000000-10_example.o"  -o "/tmp/tmpxft_0025a581_00000000-11_example_dlink.sm_30.cubin"
	fatbinary --create="/tmp/tmpxft_0025a581_00000000-7_example_dlink.fatbin" -64 -link "--image=profile=sm_30,file=/tmp/tmpxft_0025a581_00000000-11_example_dlink.sm_30.cubin" --embedded-fatbin="/tmp/tmpxft_0025a581_00000000-7_example_dlink.fatbin.c" 
	rm /tmp/tmpxft_0025a581_00000000-7_example_dlink.fatbin
	gcc -c -x c++ -DFATBINFILE="\"/tmp/tmpxft_0025a581_00000000-7_example_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"/tmp/tmpxft_0025a581_00000000-6_example_dlink.reg.c\"" -I. -D__NV_EXTRA_INITIALIZATION= -D__NV_EXTRA_FINALIZATION= -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=10 -D__CUDACC_VER_MINOR__=1 -D__CUDACC_VER_BUILD__=243 -m64 -o "/tmp/tmpxft_0025a581_00000000-12_example_dlink.o" "/usr/local/cuda/bin/crt/link.stub" 
	g++ -m64 -o "example" -Wl,--start-group "/tmp/tmpxft_0025a581_00000000-12_example_dlink.o" "/tmp/tmpxft_0025a581_00000000-10_example.o"   "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group 

clean:
	rm -rf example *.fatbin* *.cubin *.cudafe1.cpp *.ii
