for P in {20..16..-2}; do 
  eval "for TK in {$(($P))..1..-1}; 
    do tp=\$(($P*1024)); 
    tk=\$((\$TK*128*8)); 
    for r in {1..3}; 
    do 
      echo ./bin/sgemm-ttss \$tp \$tp \$tp \$tk; 
#      ./bin/sgemm-ttss \$tp \$tp \$tp \$tk;
    done; 
    echo ""; 
  done;"; 
done;
