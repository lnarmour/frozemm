CC=icc
CFLAGS=-std=c99 -I/usr/include/malloc

all: exp1 exp2

exp1: exp1.c
	$(CC) -o $@ $< $(CFLAGS)

exp2: exp2.c
	$(CC) -o $@ $< $(CFLAGS) -g

clean:
	rm -f *.o exp1 exp2
