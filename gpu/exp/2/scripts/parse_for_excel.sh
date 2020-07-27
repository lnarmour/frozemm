#!/bin/bash

if [ -z "$1" ]; then
  echo "must pass hostname as arg";
  exit 1;
fi
hostName=$1

function check_file {
  if [ ! -f $1 ]; then
    echo "log file $1 doesn't exist";
    exit 1;
  fi
}

F1_9="scripts/logs/ttss.joules.$hostName.cuda.9.1.m1000s.log";
F1_10="scripts/logs/ttss.joules.$hostName.cuda.10.1.m1000s.log";
F2_9="scripts/logs/ttss.joules.$hostName.cuda.9.1.m1024s.log";
F2_10="scripts/logs/ttss.joules.$hostName.cuda.10.1.m1024s.log";

check_file $F1_9;
check_file $F1_10;
check_file $F2_9;
check_file $F2_10;

echo "$hostName";
echo "";
echo "cuda 9.1,,cuda 10.1";
paste -d '|' <(LOG_FILE=$F1_9 python scripts/parse_ttss.py) <(LOG_FILE=$F1_10 python scripts/parse_ttss.py) | sed 's~|~,,~'
echo "";

echo "";
echo "";

echo "cuda 9.1,,cuda 10.1";
paste -d '|' <(LOG_FILE=$F2_9 python scripts/parse_ttss.py) <(LOG_FILE=$F2_10 python scripts/parse_ttss.py) | sed 's~|~,,~'
echo "";
