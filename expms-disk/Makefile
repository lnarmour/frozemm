CC=icc
CFLAGS=-std=c99 -I/usr/include/malloc

all: thruput

thruput: main.c
	$(CC) -o $@ $< $(CFLAGS)

clean:
	rm -f *.o thruput
