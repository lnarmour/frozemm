#!/bin/bash

log="tiles.log"

rm -rf ${log};

for TSJ in {24..600..16};
do
  for TSK in {24..400..16};
  do 
    count 5000 500 $TSJ $TSK >> ${log}
  done;
  for TSK in {25..400..10};
  do 
    count 5000 500 $TSJ $TSK >> ${log}
  done;
done;
