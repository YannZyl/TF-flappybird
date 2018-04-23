# -*- coding: utf-8 -*-
import numpy as np
import random
import tensorflow as tf
from collections import deque

class BrainDQN:
    def __init__(self, actions,
                 inputs_w = 80,
                 inputs_h = 80,
                 inputs_c = 4,
                 alpha=0.0001, 
                 gamma=0.99, 
                 init_epsilon = 0.05,
                 final_epsilon = 0,
                 batch_size = 32, 
                 memory_size = 50000,
                 observe_step = 100,
                 explore_step = 200000,
                 update_step = 100,
                 save_step = 10000,
                 use_double_q = True):
        self.alpha = alpha
        self.inputs_w = inputs_w
        self.inputs_h = inputs_h
        self.inputs_c = inputs_c
        self.gamma = gamma
        self.init_epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.actions = actions
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.observe_step = observe_step
        self.explore_step = explore_step
        self.update_step = update_step
        self.save_step = save_step
        self.use_double_q = use_double_q
        self.epsilon = init_epsilon
        # experience replay deque
        self.memory_replay = deque()
        self.step = 0
        # input placeholder
        with tf.name_scope('Inputs'):
            self.curr_state = tf.placeholder(tf.float32, [None,inputs_h,inputs_w,inputs_c], name='S')
            self.next_state = tf.placeholder(tf.float32, [None,inputs_h,inputs_w,inputs_c], name='S_')
            self.curr_action = tf.placeholder(tf.float32, [None,actions], name='A')
            self.q_target = tf.placeholder(tf.float32, [None,1], name='MaxQValue')
        # online Q network, update by backpropagation automatically
        with tf.name_scope('Qnetwork_online'):
            self.q_onln_values, self.q_onln_vars = self.build_qnet(self.curr_state, name='online')
        # target Q network, update by coping params from online Q network manually each t steps
        with tf.name_scope('Qnetwork_target'):
            self.q_trgt_values, self.q_trgt_vars = self.build_qnet(self.next_state, name='target')
        # optimizer for online Q network
        with tf.name_scope('Optimizer'):
            self.q_value = tf.reduce_sum(tf.multiply(self.q_onln_values, self.curr_action), axis=1, keep_dims=True)
            self.onln_loss = 0.5 * tf.reduce_mean(tf.square(self.q_target-self.q_value))
            self.optimzer = tf.train.AdamOptimizer(self.alpha, beta1=0.5).minimize(self.onln_loss, var_list=self.q_onln_vars)
        # assign op
        self.assign_op = self.copy_qnet()
        # Saver and Session
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        # initialize global variables
        self.sess.run(tf.global_variables_initializer())
    
    def build_qnet(self, inputs, name):
        # build Q network
        with tf.variable_scope(name):
            conv1 = tf.layers.conv2d(inputs, 32, 8, 4, 'SAME', activation=tf.nn.relu, name='conv1')
            pool1 = tf.layers.max_pooling2d(conv1, 2, 2, 'SAME', name='pool1')
            conv2 = tf.layers.conv2d(pool1, 64, 4, 2, 'SAME', activation=tf.nn.relu, name='conv2')
            conv3 = tf.layers.conv2d(conv2, 64, 3, 1, 'SAME', activation=tf.nn.relu, name='conv3')
            flatt = tf.layers.flatten(conv3, name='flatten')
            fc1 = tf.layers.dense(flatt, 512, activation=tf.nn.relu, name='fc1')
            qvalues = tf.layers.dense(fc1, self.actions, name='qvalues')
        # get trainable variables
        qnet_vars = tf.contrib.framework.get_trainable_variables(name)
        return qvalues, qnet_vars
    
    def copy_qnet(self):
        assign_op = []
        for onln_var, trgt_var in zip(self.q_onln_vars, self.q_trgt_vars):
            assign_op += [tf.assign(trgt_var, onln_var)]
        return assign_op
            
    def get_action(self, state):
        # check state's shape
        if len(state.shape) < 3 or len(state.shape) > 4:
            raise ValueError('Dimension of input state must be 3 or 4')
        if len(state.shape) == 3:
            state = state[np.newaxis,:]
        # compute q values at state using q target network
        q_values = self.sess.run(self.q_trgt_values, feed_dict={self.next_state: state})
        index = np.argmax(np.squeeze(q_values))
        # epsilon-greedy algorithms
        action = [0] * self.actions
        if random.random() < self.epsilon:
            index = np.random.randint(0, self.actions)
        action[index] = 1
        # epsilon decay after a train step
        if self.step > self.observe_step and self.step < self.observe_step + self.explore_step:
            self.epsilon -= (self.init_epsilon - self.final_epsilon) / self.explore_step
        return action
    
    def train_qnetwork(self, curr_state, action, reward, next_state, terminal):
        # add record into deque
        self.memory_replay.append((curr_state, action, reward, next_state, terminal))
        if len(self.memory_replay) > self.memory_size:
            self.memory_replay.popleft()
        # train network
        if self.step > self.observe_step:
            # experience replay: random sampling from deque
            batch = random.sample(self.memory_replay, self.batch_size)
            batch_S = np.array([data[0] for data in batch], dtype=np.float32)
            batch_A = np.array([data[1] for data in batch], dtype=np.float32)
            batch_R = np.array([data[2] for data in batch], dtype=np.float32)
            batch_S_ = np.array([data[3] for data in batch], dtype=np.float32)
            batch_T = np.array([data[4] for data in batch], dtype=np.bool)
            
            # get Q value of next state S_ using q target networks
            # fixed target
            next_qvals_onln, next_qvals_trgt = self.sess.run([self.q_onln_values, self.q_trgt_values], \
                                feed_dict={self.curr_state:batch_S_, self.next_state: batch_S_})
            if self.use_double_q:
                next_action = np.argmax(next_qvals_onln, axis=1)
                max_q_val = next_qvals_trgt[np.arange(next_qvals_trgt.shape[0]), next_action]
            else:
                max_q_val = np.max(next_qvals_trgt, axis=1)
            # compute gt q value
            # terminal states: gt = R
            # !termnal states: gt = R + gamma * max_a_val(next_state)
            q_target = batch_R
            idx = np.where(batch_T == False)[0]
            q_target[idx] += self.gamma * max_q_val[idx]
            
            # optimizer evaluate q network
            feed_dict = {self.curr_state:batch_S, self.curr_action:batch_A, self.q_target: q_target[:,np.newaxis]}
            self.sess.run(self.optimzer, feed_dict=feed_dict)
             
            # save model
            if self.step % self.save_step == 0:
                self.saver.save(self.sess, 'model/dqn_{}.cpkt'.format(self.use_double_q))
            # update targe q network
            if self.step % self.update_step == 0:
                self.sess.run(self.assign_op)
        
        # print information to console
        if self.step < self.observe_step:
            print('Observe state: %d, epsilon: %f' % (self.step, self.epsilon))
        elif self.step >= self.observe_step and self.step < self.explore_step + self.observe_step:
            print('Explore state: %d, epsilon: %f' % (self.step, self.epsilon))
        else:
            print('Stable state: %d, epsilon: 0' % self.step)
        
        self.step += 1