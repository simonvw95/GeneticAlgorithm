# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:20:32 2019

@author: Simon
"""
import numpy as np
import matplotlib.pyplot as plt
import time

# we will make use of a genetic algorithm in order to solve a problem


# start with defining the evaluation function
# in this case we will work with a simple bitstring of length n
# the goal is to create a bitstring of length n containing only 1's


# function to evaluate the fitness of the input vector/individual
# the fitness in this case is equal to the sum of the input
def FitnessEvaluation(int_vector):
    fitness = sum(int_vector)
    return fitness

# before we start with the body of the genetic algorithm we define the other
# necessary functions
   
    
# mutation function that flips a bit based on a mutation rate
def Mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.uniform(0, 1) < mutation_rate:
            if individual[i] == 0:
                individual[i] = 1
            else:
                individual[i] == 0
    return individual
 
               
# crossover function that combines the bitstring/chromosome at a random location
def Crossover(parent1, parent2, crossover_rate):
    if np.random.uniform(0, 1) < crossover_rate:
        cross_point = np.random.randint(0, len(parent1))
        child1 = np.concatenate((parent1[0:cross_point], parent2[cross_point:]))
        child2 = np.concatenate((parent1[cross_point:], parent2[0:cross_point]))
        return [child1, child2]
    else:
        return(parent1.copy(), parent2.copy())
    


# selection function (tournament) that selects fit individuals in the population
def TournamentSelection(population, pop_fitness):
    best_fit = 0
    # select best from k = 8 individuals 
    k = 8
    for i in range(k):
        # get a random individual
        rand_individual = np.random.randint(0, len(population))
        # compare random individuals with each other, best one wins
        if pop_fitness[rand_individual] > best_fit:
            best_fit = pop_fitness[rand_individual]
            best_individual = population[rand_individual]
            best_index = rand_individual
    return best_individual, best_fit, best_index

def GeneticAlgorithm(max_eval, pop_size, mutation_rate, crossover_rate, n):
    start_time = time.time()
    
    # initialize the population, results list and the evaluation count
    population = [0] * pop_size
    pop_fitness = [0] * pop_size
    eval_count = 0
    
    
    # initialize the population
    for i in range(pop_size):
        population[i] = np.random.randint(2, size = n)
        pop_fitness[i] = FitnessEvaluation(population[i])
        eval_count += 1
    
    # get the best individual and best fitness
    best_individual = population[np.argmax(pop_fitness)]
    best_fit = np.max(pop_fitness)
    best_list = [0] * max_eval
    
    # keep looping until we have the best possible bitstring or until we hit max evaluations
    # change this condition depending on the function evaluation, in this case the best possible fitness is equal to the size of the vector
    while best_fit != n:
        
        if eval_count >= max_eval:
            break
        
        # select 2 different individuals
        parent1, fit1, index1 = TournamentSelection(population, pop_fitness)
        parent2 = parent1
        # ensure that the second parent is not the same as the first parent
        while (parent2 == parent1).all():
            parent2, fit2, index2 = TournamentSelection(population, pop_fitness)
        
        # apply crossover and mutation
        child1, child2 = Crossover(parent1, parent2, crossover_rate)
        child1 = Mutation(child1, mutation_rate)
        child2 = Mutation(child2, mutation_rate)
        
        # evaluate the children
        fit_child1 = FitnessEvaluation(child1)
        fit_child2 = FitnessEvaluation(child2)
        
        # get the worst parent
        if fit1 >= fit2:
            best_parent = fit1
            best_index = index1
            worst_parent = fit2
            worst_index = index2
        else:
            best_parent = fit2
            best_index = index2
            worst_parent = fit1
            worst_index = index1
            
        # replace parent with child1 if child has better fitness than best parent
        if fit_child1 > best_parent:
            population[best_index] = child1
            pop_fitness[best_index] = fit_child1
            
        # if child2 has better fitness than worst parent, replace parent
        if fit_child2 > worst_parent:
            population[worst_index] = child2
            pop_fitness[worst_index] = fit_child2
        
        # update current best fitness and put it in the list
        best_fit = np.max(pop_fitness)
        best_list[eval_count - pop_size] = best_fit
        best_individual = population[np.argmax(pop_fitness)]
        
        #print("Best fitness so far: ", best_fit)
        eval_count += 1
        
    end_time = time.time()
    return [best_individual, best_fit, best_list, eval_count - pop_size, (end_time - start_time)]
        

# run the function
ind, fit, fit_list, iterations, tot_time = GeneticAlgorithm(max_eval = 10000, pop_size = 20, mutation_rate = 1/20, crossover_rate = 0.6, n = 20)

print("After ", iterations, " iterations the best individual was found after ", tot_time, " seconds")
print("The best individual looks like: ", ind, " and has a fitness of: ", fit, "\n")


# plot the results
plt.plot(range(iterations), fit_list[0:iterations], '.-')
plt.title('Results of the Genetic Algorithm')
plt.xlabel('Iterations')
plt.ylabel('Fitness of the current individuals')
plt.show()