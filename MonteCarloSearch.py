# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:21:27 2019

@author: Simon
"""
import numpy as np
import matplotlib.pyplot as plt
import time

# In order to evaluate the effectiveness of the genetic algorithm
# we can make comparisons with a monte carlo search which generates random
# configurations of the solution

# start with defining the evaluation function
# in this case we will work with a simple bitstring of length n
# the goal is to create a bitstring of length n containing only 1's



# function to evaluate the fitness of the input vector/individual
# the fitness in this case is equal to the sum of the input
def FitnessEvaluation(int_vector):
    fitness = sum(int_vector)
    return fitness



# function to generate random individuals
# max_eval is the maximum amount of evaluations
def MonteCarloSearch(max_eval, n):
    
    start_time = time.time()
    
    # initialize total result list and best result list
    fitness_list = [0] * max_eval
    best_fitness_list = [0] * max_eval
    # initialize a bit string as the current best individual
    best_individual = [0] * n
    best_fit = 0

    i = 0
    
    # keep looping until we have the best possible bitstring or until we hit max evaluations
    # change this condition depending on the function evaluation, in this case the best possible fitness is equal to the size of the vector
    while best_fit != n:
        
        if i >= max_eval:
            break
        
        # generate a random individual
        individual = np.random.randint(2, size = n)
        # evaluate that individual
        fitness_list[i] = FitnessEvaluation(individual)
        
        # if current fitness is better than previous individual, save fitness and bitstring of indidivual
        if fitness_list[i] > best_fit:
            best_fit = fitness_list[i]
            best_fitness_list[i] = best_fit
            best_individual = individual
        # if current fitness is not better, keep the previous one as best
        else:
            best_fitness_list[i] = best_fit
        
        i += 1
        
    end_time = time.time()
    return [best_individual, best_fit, fitness_list, best_fitness_list, i, (end_time - start_time)]


# run the function
ind, fit, fit_list, best_list, iterations, tot_time = MonteCarloSearch(max_eval = 10000, n = 20)

print("After ", iterations, " iterations the best individual was found after ", tot_time, " seconds")
print("The best individual looks like: ", ind, " and has a fitness of: ", fit, "\n")


# plot the results
plt.subplot(2, 1, 1)
plt.plot(range(iterations-1), fit_list[0:iterations-1], '.-')
plt.title('Results of MonteCarlo Search')
plt.xticks(range(fit))
plt.ylabel('Fitness of the current individuals')

plt.subplot(2, 1, 2)
plt.plot(range(iterations-1), best_list[0:iterations-1], 'o-')
plt.xticks(range(fit))
plt.xlabel('Iterations')
plt.ylabel('Fitness of the best individual')
plt.show()

# results can now be compared with the genetic algorithm

    