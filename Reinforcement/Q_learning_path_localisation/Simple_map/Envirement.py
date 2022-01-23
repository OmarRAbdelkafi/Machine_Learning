#coding:utf-8

import numpy as np
import random
import matplotlib.pyplot as plt # to graphics plot
import seaborn as sns # a good library to graphic plots

def maping_reward(reward_env, map_env):
    for i in range(reward_env.shape[0]):
        for j in range(reward_env.shape[1]):
            if map_env[i,j] == 1:
                reward_env[i,j] = -10
            if map_env[i,j] == 30:
                reward_env[i,j] = -1
            if map_env[i,j] == 100:
                reward_env[i,j] = 0
            if map_env[i,j] == 150:
                reward_env[i,j] = 1
            if map_env[i,j] == 255:
                reward_env[i,j] = 10
    return reward_env

def put_vertical_walls(number_of_walls, size_wall, map_env):
    for n in range(number_of_walls):
        i = random.randint(1, map_env.shape[0]-(size_wall+1))
        j = random.randint(1, map_env.shape[1]-2)
        for s in range(size_wall):
            map_env[(i+s),j] = 30

    return map_env

def put_horizontal_walls(number_of_walls, size_wall, map_env):
    for n in range(number_of_walls):
        i = random.randint(1, map_env.shape[0]-2)
        j = random.randint(1, map_env.shape[1]-(size_wall+1))
        for s in range(size_wall):
            map_env[i,(j+s)] = 30

    return map_env

def set_envirement():
    rows_size = 30
    column_size = 20
    map_env = np.zeros((rows_size,column_size))
    reward_env = np.zeros((rows_size,column_size))

    #init board and trafic lvl
    for i in range(map_env.shape[0]):
        for j in range(map_env.shape[1]):
            #create the bord
            if i == 0 or i == rows_size-1 or j == 0 or j == column_size-1:
                map_env[i,j] = 1
            else:
                #distribute low and high trafic randomly, 70% low trafic and 30% high trafic
                if random.uniform(0, 1) < 0.7:
                    #low trafic
                    map_env[i,j] = 150
                else:
                    #high trafic
                    map_env[i,j] = 100

    number_of_walls = 5
    size_wall = 4
    map_env = put_vertical_walls(number_of_walls, size_wall, map_env)
    number_of_walls = 8
    size_wall = 3
    map_env = put_horizontal_walls(number_of_walls, size_wall, map_env)

    #objectif:
    i = random.randint(1, rows_size-2)
    j = random.randint(1, column_size-2)
    map_env[i,j] = 255

    reward_env = maping_reward(reward_env, map_env)

    return map_env, reward_env

def show_envirement(map_env, reward_env):
    print(map_env)
    print(reward_env)
    sns.heatmap(map_env, cbar = False)
    plt.show()

def define_actions():
    actions = np.array([
        [-1, 0], # Up
        [1, 0], #Down
        [0, -1], # Left
        [0, 1] # Right
    ])

    return actions
