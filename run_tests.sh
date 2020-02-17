
tx=(5 10 20)

for i in ${tx[@]}; 
do 
	for j in ${tx[@]}; 
	do 
		for k in ${tx[@]}; 
		do 
			s="$(./MM.check 100 $i $j $k)";
			error="$(echo $s | grep -i error)";
			if [ -n "$error" ]; then
				echo "100 $i $j $k    $(echo $error | sed 's~.*Error\(.*\)~Error\1~')";
			fi
		done; 
	done; 
done;
