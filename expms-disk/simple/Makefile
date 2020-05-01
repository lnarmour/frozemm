CC=gcc

all: thru_4KB_1 thru_2MB_1 thru_1GB_1 thru_4KB_2 thru_2MB_2 thru_1GB_2


thru_4KB_1: exp1.c
	$(CC) -o $@ $<

thru_2MB_1: exp1.c
	$(CC) -o $@ $< -DHUGE_2MB

thru_1GB_1: exp1.c
	$(CC) -o $@ $< -DHUGE_1GB

thru_4KB_2: exp2.c
	$(CC) -o $@ $<

thru_2MB_2: exp2.c
	$(CC) -o $@ $< -DHUGE_2MB

thru_1GB_2: exp2.c
	$(CC) -o $@ $< -DHUGE_1GB



clean:
	rm -f *.o thru_*
