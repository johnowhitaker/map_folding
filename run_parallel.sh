#!/bin/bash

set -euo pipefail

# Number of parallel tasks.
mod="${MOD:-24}"

# Dimensions of the folding problem.
dimensions="${DIMENSIONS:-5 5}"

out_dir="$(mktemp -d "${TMPDIR:-/tmp}/map-folding.XXXXXX")"
trap 'rm -rf "$out_dir"' EXIT

# Run the program in parallel for each 'res' value
for (( res=0; res<mod; res++ ))
do
  echo "Starting task ${res}/${mod}"
  ./mf "${res}/${mod}" ${dimensions} > "${out_dir}/output_${res}.txt" &
done

# Wait for all background processes to finish
wait

# Aggregate the results
total=0
for (( res=0; res<mod; res++ ))
do
  count=$(cat "${out_dir}/output_${res}.txt")
  echo "Task ${res}/${mod} count: $count"
  total=$((total + count))
done

echo "Total number of foldings: $total"
