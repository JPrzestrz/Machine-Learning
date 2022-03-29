"""
Genetic Algorithms  - the minimum of 3D function searching 

"""
import map_min_search as mm
import numpy as np
import random

num_of_individuals = 100                            # number of individuals in population
                                  
num_of_epochs = 1000
p_cross = 0.5                                       # crossover probability
p_mut = 0.10                                        # mutation probability
selection_factor = 1.0                              # if higher -> higher selection (more copies of better individuals) 

num_epochs_to_show = 500                            # how frequently population must be shown - number of epochs
num_of_parameters = 2                               # number of solution parameters
L = num_of_individuals
N = num_of_parameters

Popul = np.random.rand(N,L)*20-10                   # initial population of individuals

minimum = 10e40												 # minimal function value in evolution    
record_indiv = np.array((N,1))                      # array of record solutions (individuals) during evolution


mm.show_population(Popul,"initial population")

def reproduction(Popul,fitnesses):      # new population based on fittness values
   fit_cum = np.copy(fitnesses)
   for i in range(fitnesses.size-1):    # cumulative sum
      fit_cum[i+1] += fit_cum[i]

   max_cum_value = fit_cum[fitnesses.size-1]
   Popul_new = np.copy(Popul)

   for i in range(fitnesses.size):
      rand_value = np.random.random()*max_cum_value
      for j in range(fit_cum.size):
         prev_val = 0
         if j>0:
            prev_val = fit_cum[j-1]
         if (rand_value > prev_val) & (rand_value <= fit_cum[j]):
            parent_index = j
            break
      Popul_new[:,i] = np.copy(Popul[:,j])
   return Popul_new
# end of reproduction

# crossover function 
# swaping the sequence in table 
def crossover(population,fitness):
   A_pop = population[0,:]
   B_pop = population[1,:]
   pop_cross_point = random.randint(0,len(A_pop))
   fit_cross_point = random.randint(0,len(fitness))
# end of crossover 

for ep in range(num_of_epochs):
   values_of_fun = mm.fun3(Popul[0,:],Popul[1,:])                # function values for particular individuals (points)
   fitnesses = (1.0/(values_of_fun + 2.0))**selection_factor     # fitness values (positive, as higher as quality of individual) 

   min_val = np.min(values_of_fun)
   min_index = np.argmin(values_of_fun)

   if minimum > min_val:
      print("new minimum = " + str(min_val) + " for point x1 = " + str(Popul[0,min_index]) + " x2 = " + str(Popul[1,min_index]) + "\n")
      minimum = min_val
      best_indiv = Popul[:,min_index].T 

   if ep % num_epochs_to_show == 0:
      mm.show_population(Popul,"population " + str(ep))


   # Reproduction ...
   Popul = reproduction(Popul,fitnesses)
   
   # Crossover ...
   
   # Mutation ...

   # Other operations (elitism, parameters changing etc.) ...

# end of evolution loop

#print("record path = " + str(record_indiv))
mm.show_the_best(best_indiv,"best individual, value = " + str(minimum))

