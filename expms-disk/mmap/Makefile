CC=gcc

all: thru_4KB thru_2MB thru_1GB

thru_4KB: exp1.c
	$(CC) -o $@ $<

thru_2MB: exp1.c
	$(CC) -o $@ $< -DHUGE_2MB

thru_1GB: exp1.c
	$(CC) -o $@ $< -DHUGE_1GB

clean:
	rm -f *.o thru_*
