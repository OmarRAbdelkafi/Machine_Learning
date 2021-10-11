import numpy as np
import random
import matplotlib.pyplot as plt # to graphics plot
import seaborn as sns # a good library to graphic plots

def value_function(lr, gamma, Q, st, at, New_st, New_i, New_j, reward_env):
    r = reward_env[New_i, New_j]
    best_next_action = np.argmax(Q[New_st])
    Q[st][at] = Q[st][at] + lr*(r + gamma*Q[New_st][best_next_action] - Q[st][at])

def env_finished(map_env, Q_map, st):
    i = Q_map[st, 0]
    j = Q_map[st, 1]
    return map_env[i,j] == 255

def take_action(st, Q, eps):
    # Take an action
    if random.uniform(0, 1) < eps:
        action = random.randint(0, 3)
    else: # Or greedy action
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

def QLearning(map_env, reward_env, actions):
    eps = 1
    lr = 0.1
    gamma = 0.9

    #init Q
    '''
    Q columns are actions : UP(0)/DOWN(1)/RIGHT(2)/LEFT(3) (4)
    Q rows are the states (map_env.shape[0] * map_env.shape[1])
    '''
    Q = np.zeros((map_env.shape[0] * map_env.shape[1],4))

    #map states
    Q_map = np.zeros((map_env.shape[0] * map_env.shape[1],2)).astype(int)
    for i in range(map_env.shape[0]):
        for j in range(map_env.shape[1]):
            Q_map[i*map_env.shape[1]+j,0] = i
            Q_map[i*map_env.shape[1]+j,1] = j

    #Number of learning experience
    Learning_Experiences = Q.shape[0] * Q.shape[0]
    threshold = Learning_Experiences//10

    for E in range(Learning_Experiences):

        #Update eps for exploration/exploitation balance, eps has to be more than 0.1 to not be stack in a loop greedy behaviour
        if E > threshold:
            threshold = threshold + Learning_Experiences//10
            eps = eps - 0.1
            if eps < 0.1:
                eps = 0.1

        #init car position
        i = random.randint(1, map_env.shape[0]-2)
        j = random.randint(1, map_env.shape[1]-2)

        st = i * map_env.shape[1] + j

        #If the random start is the finish point
        while env_finished(map_env, Q_map, st):
            #init car position
            i = random.randint(1, map_env.shape[0]-2)
            j = random.randint(1, map_env.shape[1]-2)

            st = i * map_env.shape[1] + j

        while not env_finished(map_env, Q_map, st):

            #choose an action following our strategy
            at = take_action(st, Q, eps)

            #Next move with at
            New_i, New_j = step(at, actions, st, Q_map, map_env)
            New_st = New_i * map_env.shape[1] + New_j

            #Update Q with value function
            value_function(lr, gamma, Q, st, at, New_st, New_i, New_j, reward_env)

            if i==New_i and j==New_j:
                Q[st][at] = Q[st][at] -1000 #out of board movement

            #New state
            st = New_st
            i = New_i
            j = New_j

    return Q, Q_map
