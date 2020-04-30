CC=icc
CFLAGS=-std=c99 -I/usr/include/malloc

all: exp2

exp2: exp2.c
	$(CC) -o $@ $< $(CFLAGS) -g

clean:
	rm -f *.o exp1 exp2
