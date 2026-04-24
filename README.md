# map_folding
Tinkering with the map/stamp folding problem

Inspired by [this video](https://www.youtube.com/watch?v=sfH9uIY3ln4) I wanted to try to calculate the next number in this sequence (https://oeis.org/A001418) - which is the number of ways an 8x8 map can be folded. 

Since this has no easy known formula, we have to do some pretty intense computation to figure it out. 

I started by converting [this Java implementation](https://github.com/archmageirvine/joeis/blob/master/src/irvine/oeis/a001/A001415.java) by Sean Irvine (which is a translation of a C version by Fred Lunnon (ALGOL68, C versions)) which implements the pseudo-code from the original 1968 paper ([PDF](https://www.ams.org/journals/mcom/1968-22-101/S0025-5718-1968-0221957-8/S0025-5718-1968-0221957-8.pdf)). o1 did the work ;)

You can speed this up by running it in parallel on multiple CPUs. To run:

- For simple use `gcc -o mf mf.c` then `./mf 5 5`
- For parallel execution, make sure compiler optimization is on `gcc -O3 -o mf mf.c` then `chmod +x run_parallel.sh` and `./run_parallel.sh` (edit to set dimensions)

The parallel case on my 12-core CPU takes about 2 minutes to compute the 6x6 case, vs about 0.8s for the 5x5 version. So 8x8 might involve quite a wait!

Update: I ran this for 7x7 and after ~42 hours I got the correct answer of `129950723279272`.

2026-04-23 update: the symmetry-reduced Modal CPU sweep replicated the 8x8 result:

```
8x8 = 162403827553180928
```

That full run used 4,804 chunks of 50,000 depth-36 symmetry-expanded cases, completed in 9,992.25s wall time, and cost about $266.75 on Modal for the main app run.

If you're running for a while, I recommend setting up some sort of notification. For e.g., I did

```
WEBHOOK_URL="https://discord.com/api/webhooks/my_webhook_url
curl -H "Content-Type: application/json" -X POST -d "{\"content\": \"$total\"}" "$WEBHOOK_URL"
```

PS: The 'parallel' bit divides up the task, but unevenly, and doesn't make sense for cases where mod > n (where n is `d x d`, e.g. for a 6x6 grid n=36). This is because it's partitioned by checking `if(mod == 0 || l_leaf != mod || m % mod == res)` where l_leaf and m range up to n, so if you try and run 100 tasks by setting mod to 100 then most of those will sit idle for a 6x6 map. I think. Smarter approach needed to parallelize this better.


## GPU folding

Grok-4 was able to write a working CUDA version that does some computation on the CPU then parallelizes the final bit. Impressive! But maybe not faster than the CPU case. To compile: `nvcc -O3 -o gpu_folding gpu_folding_try1.cu -std=c++11`. To run: `time ./gpu_folding 5 15` (partition_depth of 15 means it finds ~13k partial states that it ships off to the GPU to do) - takes about 0.6s. `time ./gpu_folding 6 18` took like 11 minutes. Keeping for reference. Bummer it's slow...

No worries though! I had o3 do code review, gave some suggestions back to grok 4 and had it re-write it to be more efficient. `nvcc -O3 -o gpu_folding gpu_folding.cu -std=c++11`. CPU step on `time ./gpu_folding 6 18` was near-instant and total time was 1m24! `./gpu_folding 7 25` used up all my RAM so there are improvements to be made, but `7 23` worked and after keeping my 3090 huming for XXX I got the correct answer.

Finally, I had it attempt a multi-GPU solution - which on  GPUs did `6 18` in 20s - extremely promising!
`modal launch jupyter --gpu A10G:4 --image 'nvidia/cuda:12.8.0-devel-ubuntu22.04'` (can also use vscode instead of jupyter) to get an instance quickly for testing. 

It's wild to me that AI can do this stuff.

### How It Works (grok 4 summary)

This implementation computes the number of ways to fold an n x n sheet of stamps using a backtracking algorithm adapted from the original 1968 paper by Lunnon. The core is a depth-first search (DFS) that builds valid foldings by placing "leaves" (stamps) while ensuring folding constraints via precomputed coordinate mappings (arrays c and d).

To accelerate computation on GPUs:

Precomputation on CPU: We run the backtracking up to a user-specified partition_depth (e.g., 20-30), collecting partial states (configurations of arrays a, b_arr, and gapter at that depth). This partitions the enormous search space into independent sub-trees. For larger n, this generates millions of states quickly.
GPU Parallelism: The partial states are transferred to GPU memory. Each GPU thread resumes backtracking from one partial state to completion, counting valid foldings (myCount += n for each complete folding). Results are atomically summed into a global counter.
Optimizations: Large arrays (bigP, c, d) are moved to CUDA constant memory for fast, read-only access, reducing per-thread local memory (~70-80 KB savings) and boosting occupancy (10% → 60-80%). The CPU phase skips gap computation beyond the partition depth for speed.

## Symmetry-reduced solver

`sym_fold.cpp` is the current fastest path in this repo. It is an attributed C++ port of gsitcia's Code Golf Stack Exchange symmetry-reduced solver, using center-out spiral numbering, D4 square symmetries, and stack reversal to reduce the search space before counting.

Build and run locally:

```
clang++ -O3 -march=native -std=c++20 -pthread -o sym_fold sym_fold.cpp
./sym_fold 6 12
./sym_fold 7 12
```

Useful 8x8 chunk/sampling mode:

```
MF_STREAM_CASES=1 MF_EXPANDED_N=36 ./sym_fold 8 12 100000 0
```

For Modal CPU chunks:

```
modal run modal_sym.py::main --n 8 --threads 16 --expanded-n 36 --start-case 0 --max-cases 100000
```

For the full replicated 8x8 sweep:

```
modal run modal_sym.py::sweep --n 8 --threads 16 --expanded-n 36 \
  --total-cases 240192829 --chunk-size 50000 \
  --out results/8x8-depth36-full.jsonl
```

If a sweep is interrupted, rerun with `--resume-existing` against the same output file to skip chunk starts that are already present. See `docs/8x8-method.md` for the method writeup. The experiment log in `EXPERIMENTS.md` has timings, validation details, and partial chunk results. Speculative 9x9 feasibility work lives under `experiments/9x9/`.
