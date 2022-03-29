"""
   Simulated Annealing  - the minimum of 2D function searching

"""
import map_min_search as mm
import numpy as np
import get_point as gp
import math
num_of_steps = 2000    
for s in range(50,100):
   np.random.seed(s)
   for t in [1,2,5]:
      for dt in [0.15,0.2]:
         for const in [0.5,0.9,5]:
            Average = 0 
            text=""
            min_val = 10
            for ix in range(10):
               # number of solution parameters
               num_of_parameters = 2                               
               N = num_of_parameters
               # temperature (randomness coefficient)
               T = t                                          
               # minimal temperature
               T_min = 0.00000001                                        
               # change of temperature
               wT = dt                                            
               # constant due to the influence of T for acceptance probability
               c = const                                            

               # initial solution - random point
               Solution = np.random.rand(N)*20-10

               E_min = 10e40							            # minimal function value
               E_prev = 0                                          # previous value of the function
               Records = np.empty((0,N))                           # array of record solutions
               #np.random.seed(ix)
               for ep in range(num_of_steps):
                  SolutionNew = gp.GetPointInCircle(Solution,T,-10,10)                                # new solution (should be near previous one !)

                  E = mm.fun3(SolutionNew[0],SolutionNew[1])       # function value for point coordinates

                  dE = E - E_prev                                  # change of function value (dE < 0 means than new solution is better)

                  # acceptance probability
                  if dE < 0:
                     p_accept = 1
                  else:
                     p_accept = math.exp(-dE/(T))
                  p_accept = math.exp(-dE/(T))

                  if np.random.rand() < p_accept:
                     Solution = SolutionNew
                     E_prev = E

                  if E_min > E:
                     #print("new minimum = " + str(E) + " for point x1 = " + str(SolutionNew[0]) + " x2 = " + str(SolutionNew[1]) + "\n")
                     E_min = E
                     Solution_min = SolutionNew
                     Records = np.append(Records, [SolutionNew], axis = 0)

                  # temperature changing (can be only after accaptance or in another place)
                  T = np.random.rand()*10 * wT                                      
                  if T < T_min:
                     T = T_min
               # end of steps loop
               text += "best solution, value = " + str(E_min) + " for point x1 = " + str(Solution_min[0]) + " x2 = " + str(Solution_min[1]) +"\n"
               Average +=E_min
               if E_min > min_val:
                  min_val = E_min
               print(text)
               #mm.show_point_sequence(Records,"record sequence, " + text)
            Average = Average/10
            if min_val <= -1.8:
               print(text)
               print('Average with(T={}, wT={}, c={}, seed={}) is {}'.format(t,dt,const,s,Average))