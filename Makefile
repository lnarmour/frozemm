CC=icc
DEFS= 
OPTS=-O3 -xcore-avx2
CFLAGS=-std=c99 -I/usr/include/malloc/ $(OPTS)

all: MM

MM: ss.c
	$(CC) ss.c -o MM $(CFLAGS) $(DEFS) -qopt-report-phase=vec -qopt-report=5

clean:
	rm -f *.o MM
