#!/bin/bash
set -x

cd ~/ISCA2012/OpenCLKernelScheduler/bin/linux/release

for MULTSIZE in 1 2 3 4 5 6 7 8 9 10
do
  make -C ../../.. clean
  make -C ../../.. MULTSIZE=$MULTSIZE || exit
  for iteration in `seq 1 10`
  do
    echo "0 0" | ./scheduler
  done
done