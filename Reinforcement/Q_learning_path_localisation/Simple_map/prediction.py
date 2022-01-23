#coding:utf-8

import numpy as np
import random
import matplotlib.pyplot as plt # to graphics plot
import seaborn as sns # a good library to graphic plots

def env_finished(map_env, Q_map, st):
    i = Q_map[st, 0]
    j = Q_map[st, 1]
    return map_env[i,j] == 255

def take_greedy_action(st, Q):
    action = np.argmax(Q[st])
    return action

def step(at, actions, st, Q_map, map_env):
    """
        Action: 0, 1, 2, 3 : Up, Down, Right, left
    """
    i = Q_map[st, 0]
    j = Q_map[st, 1]

    i = max(0, min(i + actions[at][0], map_env.shape[0]-1))
    j = max(0, min(j + actions[at][1], map_env.shape[1]-1))

    return i, j

def Make_prediction(start_i, start_j, Q, Q_map, map_env, actions):

    #init car position
    i = start_i
    j = start_j

    if map_env[i,j] != 255:
        map_env[i,j] = 230

    way = [[i,j]]

    st = i * map_env.shape[1] + j

    while not env_finished(map_env, Q_map, st):
        #choose an action following our strategy
        at = take_greedy_action(st, Q)

        #Next move with at
        New_i, New_j = step(at, actions, st, Q_map, map_env)

        if map_env[New_i,New_j] != 255:
            map_env[New_i,New_j] = 230

        if i==New_i and j==New_j:
            Q[st][at] = Q[st][at] - 1000 #to avoid out of board movement
        if [New_i,New_j] in way:
            Q[st][at] = Q[st][at] - 1000 #to avoid loop mvt

        way.append([New_i,New_j])
        New_st = New_i * map_env.shape[1] + New_j

        #New state
        st = New_st
        i = New_i
        j = New_j

    print(way)
    sns.heatmap(map_env, cbar = False)
    plt.show()
