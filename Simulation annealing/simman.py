"""
   Simulated Annealing  - the minimum of 2D function searching

"""
import map_min_search as mm
import numpy as np
import math
                                  
num_of_steps = 2000                                 # number of steps: do not change 2000

num_of_parameters = 2                               # number of solution parameters
N = num_of_parameters

T = 10 # temperature (randomness coefficient)
T_min = 0.000001 # minimal temperature
wT = 1 # change of temperature
c = 2 # constant due to the influence of T for acceptance probability

Solution = np.random.rand(N)*20-10                  # initial solution - random point


E_min = 10e40							            # minimal function value
E_prev = 0                                          # previous value of the function
Records = np.empty((0,N))                           # array of record solutions

# mm.show_the_point(Solution,"initial solution")

def GetPointInCircle(r):
    circle_r, x0, y0 = r, 0, 0
    alpha = 2 * math.pi * np.random.rand()
    r = circle_r * math.sqrt(np.random.rand())
    x = r * math.cos(alpha) + x0
    y = r * math.sin(alpha) + y0
    Point = [x,y]
    return Point

for ep in range(num_of_steps):

    SolutionNew = Solution + GetPointInCircle(T) # new solution (should be near previous one !)
    while(SolutionNew[0]>10 or SolutionNew[0]<-10 or SolutionNew[1]<-10 or SolutionNew[1]> 10):
        SolutionNew = Solution + GetPointInCircle(T)

    E = mm.fun3(SolutionNew[0],SolutionNew[1])       # function value for point coordinates
    dE = E - E_prev                                  # change of function value (dE < 0 means than new solution is better)

    p_accept = 1 + (1 / (math.exp(dE / (c * T))))                                  # acceptance probability
    
    if((ep>995 and ep<=1000) or (ep>1995)):
        print("Step: " + str(ep+1))
        print("Actual Point: " + str(Solution) + " New Point: " + str(SolutionNew) + " Value delta: " + str(dE) +
        " Accept prob: " + str(p_accept))
        print("New T: "+ str(T))

    if np.random.rand() < p_accept:
        Solution = SolutionNew
        E_prev = E

    if E_min > E:
        print("new minimum = " + str(E) + " for point x1 = " + str(SolutionNew[0]) + " x2 = " + str(SolutionNew[1]) + "\n")
        E_min = E
        Solution_min = SolutionNew
        Records = np.append(Records, [SolutionNew], axis = 0)

    T = (np.random.rand()*10)* wT # temperature changing (can be only after accaptance or in another place)
    wT -= 0.00055
    if T < T_min:
        T = T_min
# end of steps loop

text = "\nbest solution, value = " + str(E_min) + " for point x1 = " + str(Solution_min[0]) + " x2 = " + str(Solution_min[1])
print(text + "\n")
mm.show_point_sequence(Records,"record sequence, " + text)

