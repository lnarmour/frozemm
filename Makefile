CC=icc
PRECISION=SINGLE
DEFS= 
OPTS= -O3 -xcore-avx2
CFLAGS=  -std=c99  -I/usr/include/malloc/ ${OPTS}
OBJS=ss.o

all: MM

MM: $(OBJS) ss-wrapper.c
	$(CC) ss-wrapper.c $(OBJS) -o MM $(CFLAGS) -D$(PRECISION)=1

ss.o: ss.c
	$(CC) ss.c -c -o ss.o $(CFLAGS) $(DEFS) -D$(PRECISION)=1 -ftree-vectorize -mavx2 -fopt-info-all=opt.info

clean:
	rm -f *.o MM
