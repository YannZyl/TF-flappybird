# -*- coding: utf-8 -*-
import cv2
import numpy as np
from dqn_tf import BrainDQN
from game.flappy_bird import FlappyBirdGame

def preprocess(observation):
    # binary format, and resize to (80,80)
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    # remove mean
    observation -= 128
    # reshape to (80,80,1)
    return np.reshape(observation,(80,80,1))

def playFlappyBird():
    num_actions = 2
    dqn = BrainDQN(num_actions)
    game = FlappyBirdGame()
    init_state = np.array([1,0], dtype=np.float32)
    observation, reward, terminal = game.frame_step(init_state)
    observation = preprocess(observation)
    # generate first batch 
    curr_state = np.concatenate((observation,observation,observation,observation), axis=2)
    # train DQN
    while True:
        # get action at current state
        action = dqn.get_action(curr_state)
        # get next state, reward
        next_observation, reward, terminal = game.frame_step(action)
        next_observation = preprocess(next_observation)
        next_state = np.append(curr_state[...,1:], next_observation, axis=2)
        # train DQN
        dqn.train_qnetwork(curr_state, action, reward, next_state, terminal)
        
        
playFlappyBird()