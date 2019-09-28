# GeneticAlgorithm
Quick python implementation of a genetic algorithm for a simple bitstring problem, framework can be copied and altered to match different problems.

A very simple implementation of a genetic algorithm in python, the optimization problem used in this example is maximizing the amount of 1's in a bitstring of length n, the fitness function is simply thus the sum of the entire bitstring (e.g. the largest fitness value of a bitstring of length 10 is 10: bitstring = [1,1,1,1,1,1,1,1,1,1], fitness = sum(bitstring) = 10

Additionally, a MonteCarloSearch function is included which allows for easy comparisons of the effectiveness of a genetic algorithm vs a random search. 

The parameters of each function are explained in the comments.
