# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 20:45:19 2021

@author: Stephan
"""

from IPython.display import display

# Define all states
all_states=["Hungry_Tired", "Hungry_Rested", "Full_Tired", "Full_Rested"]
#actions
actions = ['rest', 'hunt']
# Initialize q value
Q= {}
for s in all_states:
    for a in actions:
        Q[s,a] = 0
        
def q_learning(iteration, reward, s, a, s_nxt):
    max_qnxt = max(Q[s_nxt,'hunt'],Q[s_nxt,'rest'])
    Q[(s,a)] = Q[(s,a)] + learning_rate * (reward + GEMMA * max_qnxt - Q[(s,a)])
    
    print(iteration)
    display(Q)

# Define parameters
GEMMA = 1
learning_rate = .5

q_learning(1, 1,"Hungry_Tired", 'rest', "Hungry_Rested")
q_learning(2, 0,"Hungry_Tired", 'hunt', "Hungry_Tired")
q_learning(3, 1, "Hungry_Rested", "rest", "Hungry_Rested")
q_learning(4, 2, "Hungry_Rested", "hunt", "Full_Rested")
q_learning(5, 1, "Full_Rested","rest", "Hungry_Rested")
q_learning(6, 0, "Hungry_Rested", "hunt", "Hungry_Tired")
