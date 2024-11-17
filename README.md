# map_folding
Tinkering with the map/stamp folding problem

Inspired by [this video](https://www.youtube.com/watch?v=sfH9uIY3ln4) I wanted to try to calculate the next number in this sequence (https://oeis.org/A001418) - which is the number of ways an 8x8 map can be folded. 

Since this has no easy known formula, we have to do some pretty intense computation to figure it out. 

I started by converting [this Java implementation](https://github.com/archmageirvine/joeis/blob/master/src/irvine/oeis/a001/A001415.java) by Sean Irvine (which is a translation of a C version by Fred Lunnon (ALGOL68, C versions)) which implements the pseudo-code from the original 1968 paper ([PDF](https://www.ams.org/journals/mcom/1968-22-101/S0025-5718-1968-0221957-8/S0025-5718-1968-0221957-8.pdf)). o1 did the work ;)

You can speed this up by running it in parallel on multiple CPUs. To run:

- For simple use `gcc -o mf mf.c` then `./mf 5 5`
- For parallel execution, make sure compiler optimization is on `gcc -O3 -o mf mf.c` then `chmod +x run_parallel.sh` and `./run_parallel.sh` (edit to set dimensions)

The parallel case on my 12-core CPU takes about 2 minutes to compute the 6x6 case, vs about 0.8s for the 5x5 version. So 8x8 might involve quite a wait!

Update: I ran this for 7x7 and after ~42 hours I got the correct answer of `129950723279272`. Who will do 8x8?
