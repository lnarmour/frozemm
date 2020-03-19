BINS=(MM.novec MM.avx2 MM.500.56.56.novec MM.500.56.56.avx2)
for b in ${BINS[@]}; do export BIN="$b"; m="$(cat tmp/$(hostname) | grep -A3 "$BIN" | grep '^[0-9]\+' | python3 utilities/mean.py)" && printf "$m,"; done;
printf ",,,";
for b in ${BINS[@]}; do export BIN="$b"; s="$(cat tmp/$(hostname) | grep -A3 "$BIN" | grep '^[0-9]\+' | python3 utilities/std.py)" && printf "$s,"; done; echo "";
