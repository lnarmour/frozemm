CC = gcc
#CFLAGS = -O3 -DSTREAM_ARRAY_SIZE=512000000 -fopenmp -opt-streaming-stores=always -opt-prefetch-distance=64,8 -opt-streaming-cache-evict=0

N = 512000000
CFLAGS = -O3 -fopenmp -DSTREAM_ARRAY_SIZE=$(N)

all: stream_c.exe

stream_c.exe: stream.c
	$(CC) $(CFLAGS) stream.c -o stream_c.exe

clean:
	rm -f stream_f.exe stream_c.exe *.o

# an example of a more complex build line for the Intel icc compiler
stream.icc: stream.c
	icc -O3 -xCORE-AVX2 -ffreestanding -qopenmp -DSTREAM_ARRAY_SIZE=80000000 -DNTIMES=20 stream.c -o stream.omp.AVX2.80M.20x.icc
