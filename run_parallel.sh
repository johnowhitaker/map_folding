#!/bin/bash

# Number of parallel tasks (modify as needed)
mod=24

# Dimensions of the folding problem (modify as needed)
dimensions="5 5"

# Remove any existing output files
rm -f output_*.txt

# Run the program in parallel for each 'res' value
for (( res=0; res<mod; res++ ))
do
  echo "Starting task ${res}/${mod}"
  ./mf "${res}/${mod}" ${dimensions} > "output_${res}.txt" &
done

# Wait for all background processes to finish
wait

# Aggregate the results
total=0
for (( res=0; res<mod; res++ ))
do
  count=$(cat "output_${res}.txt")
  echo "Task ${res}/${mod} count: $count"
  total=$((total + count))
done

echo "Total number of foldings: $total"
