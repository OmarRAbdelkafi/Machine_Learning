#coding:utf-8

import numpy as np

import Simple_map.Envirement as SM_E
import Simple_map.Algorithm as SM_A
import Simple_map.prediction as SM_P


"""
1. Problem description :
* The envirement is a grid map with gray lvl between 0 and 255 :
- The location to reach: lvl 255 / reward: 10
- The board: lvl 1 / reward: -10
- The wall: lvl 30 / reward: -1
- The way with low trafic: lvl 150 / reward: 1
- The way with high trafic: lvl 100 / reward: 0
--> The car experience can be represented with lvl 230

2. Objectif:
Learn the best way to reach the location
"""

"""
Input data:
- None
"""

"""
Read data:
-None
"""

"""
Create envirement and actions
"""
map_env, reward_env = SM_E.set_envirement()
SM_E.show_envirement(map_env, reward_env)
actions = SM_E.define_actions()

"""
Reinforcement learning : Qlearning
"""
Q, Q_map = SM_A.QLearning(map_env, reward_env, actions)
print(Q)

"""
Prediction
"""
start_i = 1
start_j = 1
SM_P.Make_prediction(start_i, start_j, Q, Q_map, map_env, actions)
