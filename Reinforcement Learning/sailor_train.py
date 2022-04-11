import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf
import random

number_of_episodes = 4000                   # number of training epizodes (multi-stage processes) 
gamma = 1.0                                 # discount factor
alpha = 0.1                                 # training speed (factor between 0 and 1) 
epsilon = 0.6                              # exploration factor (between 0 and 1)

#file_name = 'map_small.txt'
#file_name = 'map_easy.txt'
file_name = 'map_spiral.txt'
#file_name = 'map_big.txt'

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(2.5*(num_of_rows + num_of_columns))    # maximum number of steps in an episode
Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)  # trained usability table of <state,action> pairs
sum_of_rewards = np.zeros([number_of_episodes], dtype=float)
table1 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
table = [0.1]
for alpha in table:
    for epsilon in [0.6]:
        sid = random.randint(1,10)
        random.seed(sid)
        np.random.seed(sid)
        print('seed {}'.format(sid))

        expl = 0
        Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)  # trained usability table of <state,action> pairs
        sum_of_rewards = np.zeros([number_of_episodes], dtype=float)
        for episode in range(number_of_episodes):
            # state = np.zeros([2],dtype=int)    
            # initial state here [1 1] but rather random due to exploration
            state = np.array([random.randint(0,num_of_rows-1),0],dtype=int)
            if np.random.random() < 0.5 :
                state=[random.randint(1,num_of_rows-1),random.randint(1,num_of_columns-1)]

            #print('initial state = ' + str(state) )
            the_end = False
            nr_pos = 0
            #reward_map_curr = reward_map
            while the_end == False:
                nr_pos = nr_pos + 1;                            # move number
            
                # Action choosing (1 - right, 2 - up, 3 - left, 4 - bottom): 
                if random.random() > epsilon and expl>1999:
                    action = np.argmax(Q[state[0],state[1]])+1
                else:
                    action = random.randint(1,4)
                    expl+=1

                state_next, reward  = sf.environment(state, action, reward_map); 
                #print('state next[1] = {}'.format(state_next[1]))

                # State-action usability modifcication: TODO
                Q[state[0],state[1],action-1] = Q[state[0],state[1],action-1] + (alpha * (reward + (gamma * np.max(Q[state_next[0],state_next[1]]))- Q[state[0],state[1],action-1]))
                #print('state = ' + str(state) + ' action = ' + str(action) +  ' -> next state = ' + str(state_next) + ' reward = ' + str(reward))
                state = state_next;      # going to the next state
            
                # end of episode if maximum number of steps is reached or last column
                # is reached
                if (nr_pos == num_of_steps_max) | (state[1] >= num_of_columns-1):
                    the_end = True;                                  
            
                sum_of_rewards[episode] += reward
            #if episode % 500 == 0:
                #print('episode = ' + str(episode) + ' average sum of rewards = ' + str(np.mean(sum_of_rewards)))
        #print('average sum of rewards = ' + str(np.mean(sum_of_rewards)))
        print('alpha={}, epsilon={}'.format(alpha,epsilon))
        sf.sailor_test(reward_map, Q, 1000)
        sf.draw(reward_map,Q)

#sf.draw(reward_map,Q)
