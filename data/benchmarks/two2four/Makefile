LIBRARIES=-lm
CC=gcc
LD=xild
AR=xiar
CFLAGS=  -std=c99  -I/usr/include/malloc/ -fopenmp -O3
all: ttf

debug: CFLAGS =-DDEBUG -O0 -g -Wall -Wextra -std=c99 -I/usr/include/malloc/ ${MKL_FLAGS}
debug: ttf

ttf: main.c
	$(CC) main.c -o ttf $(CFLAGS) $(LIBRARIES)

clean:
	rm -f *.o ttf
