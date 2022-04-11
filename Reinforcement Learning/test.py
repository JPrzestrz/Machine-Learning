from random import randint
import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf
import random

number_of_episodes = 4000                   # number of training epizodes (multi-stage processes) 
gamma = 1.0                                 # discount factor

alpha = 0.1                                 # training speed factor
epsilon = 0.6
counter=0                               # exploration factor

#file_name = 'map_easy.txt'
#file_name = 'map_spiral.txt'
file_name = 'map_big.txt'

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

for seed in range(18,20):
    random.seed(seed)
    np.random.seed(seed)

    num_of_steps_max = int(2.5*(num_of_rows + num_of_columns))    # maximum number of steps in an episode
    Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)  # trained usability table of <state,action> pairs
    sum_of_rewards = np.zeros([number_of_episodes], dtype=float)

    for episode in range(number_of_episodes):
        state = np.zeros([2],dtype=int)                            # initial state here [1 1] but rather random due to exploration
        if np.random.random() < 0.5 :
            state=[randint(1,num_of_rows-1),randint(1,num_of_columns-1)]

        #print('initial state = ' + str(state) )
        the_end = False
        nr_pos = 0
        #reward_map_curr = reward_map
        while the_end == False:
            nr_pos = nr_pos + 1;                            # move number
        
            # Action choosing (1 - right, 2 - up, 3 - left, 4 - bottom): 
            if np.random.random() < epsilon and counter > 2000:
                temp = np.argmax(Q[state[0],state[1]])
                action=temp+1
            else:
                counter+=1
                action=randint(1,4)
            state_next, reward  = sf.environment(state, action, reward_map); 
        
            # State-action usability modifcication:
            #Q[state[0],state[1],action-1] = ........................................
            #old_q=Q[state[0],state[1],action-1]
            #diff=reward + (gamma * np.max(Q[state_next[0],state_next[1]])) - old_q
            #new_q=old_q + (alpha * diff)
            #Q[state[0],state[1],action-1]=new_q
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
    sf.sailor_test(reward_map, Q, 1000)
    sf.draw(reward_map,Q)

#sf.sailor_test(reward_map, Q, 1000)
#sf.draw(reward_map,Q)
