# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 22:47:38 2021

@author: Stephan
"""
# https://gist.github.com/francoisstamant/685913ea116be8a9e95974a239c1d5a9#file-value_iteration_2-py
import numpy as np
import pandas as pd
from IPython.display import display
'''
==================================================
Initial set up
==================================================
'''
# Define all states
all_states=["Hungry_Tired", "Hungry_Rested", "Full_Tired", "Full_Rested"]
# Define reward with order of all states
rewards_list = [0.,1.,1.,2.]
# Ddfine transition probabilities to next state
d = {}
d[("Hungry_Tired", 'rest')] = [0,1,0,0]
d[("Hungry_Rested",'rest')] = [0,1,0,0]
d[("Full_Tired",'rest')] = [0,1,0,0]
d[("Full_Rested",'rest')] = [0,1,0,0]
d[("Hungry_Tired", 'hunt')] = [.8,0,.2,0]
d[("Hungry_Rested",'hunt')] = [.2,0,0,.8]
d[("Full_Tired",'hunt')] = [.8,0,.2,0]
d[("Full_Rested",'hunt')] = [.2,0,.8,0]
transition = pd.DataFrame(data = d)
#actions
actions = ['rest', 'hunt']
# Define initial value 
value_list_old = np.zeros((4,))
value_list_new = np.zeros((4,)) 
# Empty q table
Q = {}

#==================================================
#Value Iteration
#==================================================

iteration = 0
max_inter = 4
GEMMA = 1
result = pd.DataFrame(value_list_old).transpose()
while iteration < max_inter:

    for i, s in enumerate(all_states):
        for a in actions:
            # find the q value for each state-action pair
            immidate_reward = np.multiply(rewards_list,transition[(s,a)]).sum()
            future_value = np.multiply(value_list_old,transition[(s,a)]).sum()
            Q[(s,a)] = immidate_reward + GEMMA * future_value
            
        # find the state value
        value_list_new[i] = max(Q[s,'hunt'],Q[s,'rest'])
        
    value_list_old = value_list_new.copy() # upate value
    # update value result
    result = result.append(pd.DataFrame(value_list_old).transpose(), ignore_index=True)   
    # print q values
    print(iteration + 1)
    display(Q)
    
    iteration += 1
result.columns = all_states
print(result)