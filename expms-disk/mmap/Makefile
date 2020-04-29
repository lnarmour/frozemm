CC=gcc

all: thru_4KB_reads thru_2MB_reads thru_1GB_reads thru_4KB_writes thru_2MB_writes thru_1GB_writes

thru_4KB_reads: exp1.c
	$(CC) -o $@ $< -DREADS

thru_2MB_reads: exp1.c
	$(CC) -o $@ $< -DREADS -DHUGE_2MB

thru_1GB_reads: exp1.c
	$(CC) -o $@ $< -DREADS -DHUGE_1GB


thru_4KB_writes: exp1.c
	$(CC) -o $@ $<

thru_2MB_writes: exp1.c
	$(CC) -o $@ $< -DHUGE_2MB

thru_1GB_writes: exp1.c
	$(CC) -o $@ $< -DHUGE_1GB



clean:
	rm -f *.o thru_*
