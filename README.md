# sim-anneal
Implementation of simulated annealing for jigsaw puzzle solving purposes.

## Algorithm

* https://en.wikipedia.org/wiki/Simulated_annealing

## Sampling

* https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm

## Energy

The energy function is the square root of the average of the squared differences around the tile edges.

## Proposal distribution

The proposal distribution prioritizes tiles with higher energy according to a normal distribution.


