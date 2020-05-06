CC=gcc -O0

all: two2four_4KB two2four_2MB two2four_1GB

two2four_4KB: two2four.c
	$(CC) -o $@ $<

two2four_2MB: two2four.c
	$(CC) -o $@ $< -DHUGE_2MB

two2four_1GB: two2four.c
	$(CC) -o $@ $< -DHUGE_1GB

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

thru_4KB_3: exp3.c
	$(CC) -o $@ $<

thru_2MB_3: exp3.c
	$(CC) -o $@ $< -DHUGE_2MB

thru_1GB_3: exp3.c
	$(CC) -o $@ $< -DHUGE_1GB



clean:
	rm -f *.o thru_* two2four_*
