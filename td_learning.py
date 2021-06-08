# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 20:09:14 2021

@author: Stephan
"""
import numpy as np
import pandas as pd
from IPython.display import display

V_Hungry_tired = 0
learning_rate = 0.5
GEMMA = 1

# Define all states
all_states=["Hungry_Tired", "Hungry_Rested", "Full_Tired", "Full_Rested"]
V = [0.,0.,2., 0.]
# temporal difference

def value(immidiate_reward, next_state, interation):
    A_k_1 = V[0]
    nxt_state = all_states.index(next_state)
    future_value = V[nxt_state]
    V_k = immidiate_reward + GEMMA * future_value
    A_k = A_k_1 + learning_rate * (V_k - A_k_1)
    print(interation, "iteration:")
    print("Ak-1:{} Vk:{} Vnxt:{} Ak:{}".format(A_k_1, V_k, future_value,A_k))
    V[0] = A_k
    
value(0, "Hungry_Tired", 1)
value(1, "Full_Tired", 2)
value(1, "Full_Tired", 3)
value(1, "Full_Tired", 4)
