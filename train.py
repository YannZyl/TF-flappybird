# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from dqn_tf import BrainDQN
import matplotlib.pyplot as plt
from game.flappy_bird import FlappyBirdGame
from sklearn.externals import joblib

def preprocess(observation):
    # binary format, and resize to (80,80)
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    # remove mean, must transform observation from uint8 to int8/float32 before remove mean!!!
    observation = np.array(observation,dtype=np.float32) - 128
    #observation -= 128
    # reshape to (80,80,1)
    return np.reshape(observation,(80,80,1))

def playFlappyBird(use_double_q=False):
    num_actions = 2
    dqn = BrainDQN(num_actions, use_double_q=use_double_q)
    game = FlappyBirdGame()
    init_state = np.array([1,0], dtype=np.float32)
    observation, reward, terminal = game.frame_step(init_state)
    observation = preprocess(observation)
    # generate first batch 
    curr_state = np.concatenate((observation,observation,observation,observation), axis=2)
    records = []
    score = 0
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
        # update current state
        curr_state = next_state
        # update score
        score += reward
        # update record, clear score to start next episode
        if terminal:
            records.append(score)
            score = 0
        # save every t episode
        if terminal and len(records) % 10 == 0 and len(records) > 0:
            if os.path.exists('output/records.pkl'):
                os.remove('output/records.pkl')
            joblib.dump(records, 'output/records.pkl')


playFlappyBird()
records = joblib.load('output/records.pkl')
plt.plot(np.arange(len(records)), records, 'b')
plt.show()