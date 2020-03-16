
tx=(5 10 20 25 50)

for i in ${tx[@]}; 
do 
	for j in ${tx[@]}; 
	do 
		for k in ${tx[@]}; 
		do 
			s="$(./MM.check 100 $i $j $k)";
			seg="$(echo $s | grep -i seg)";
			if [ -n "$error" ]; then
				echo "seg faults: 100 $i $j $k";
				exit 1;
			fi	
			error="$(echo $s | grep -i error)";
			if [ -n "$error" ]; then
				echo "100 $i $j $k    $(echo $error | sed 's~.*Error\(.*\)~Error\1~')";
			fi
		done; 
	done; 
done;
