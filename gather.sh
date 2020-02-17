#!/bin/bash

make clean > /dev/null 2>&1
make > /dev/null 2>&1

N=8192
TK=(8192 4096 2048 1024 512 256 128 64 32 16)
printf "N,tk:time0,time1,...\n"
for tk in ${TK[@]};
do
	printf "$N,$tk:"
	gflopss="$(./MM $N $N $N $tk | grep Effective | sed 's~.*: \(.*\) gflo.*~\1~')"
	printf "$gflopss"
	for run in {2..5};
	do
		gflopss="$(./MM $N $N $N $tk | grep Effective | sed 's~.*: \(.*\) gflo.*~\1~')"
		printf ",$gflopss"
	done;
	printf "\n"
done;
