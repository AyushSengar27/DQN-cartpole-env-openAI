#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import time
from myDQNagent import DQN_learning

from Helper import LearningCurvePlot, smooth

    
def average_over_repetitions(enable_experience_replay, enable_target_network,n_eval_episodes,eval_interval, max_epoch_env_steps,n_repetitions, num_epochs, lr, discount_factor, mode, eps_start, smoothing_window):
    returns_over_repetitions = []
    now = time.time()
    for repetition in range(n_repetitions):
        
        print("Repetition Number: ", repetition + 1)
        learning_curve,timesteps = DQN_learning(num_epochs=num_epochs,max_epoch_env_steps=max_epoch_env_steps,eval_interval=eval_interval,n_eval_episodes=n_eval_episodes,
                                      lr=lr,discount_factor=discount_factor,eps_start=eps_start,enable_target_network=enable_target_network, 
                                      enable_experience_replay=enable_experience_replay,mode=mode)
        returns_over_repetitions.append(learning_curve)
    
    # Average the learning curves across repetitions
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    learning_curve = np.mean(np.array(returns_over_repetitions),axis=0) # average over repetitions  
    if smoothing_window is not None: 
        learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve, timesteps  

def experiment():

    # #### Assignment 5: DQN Variations
    n_repetitions = 1
    num_epochs = 200
    max_epoch_env_steps=150
    eval_interval=5
    n_eval_episodes=10
    lr=0.001
    discount_factor=0.95
    
    eps_start=0.01
    mode='epsilon-greedy'
    smoothing_window = 9
    plot = False

    Variations = [('DQN', True, True), ('DQN - ER', False, True), ('DQN - TN', True, False), ('DQN - ER - TN', False, False)]
    
    
    Plot = LearningCurvePlot(title='DQN Variations')    
    Plot.set_ylim(0, 300)
    
    for label, enable_experience_replay, enable_target_network in Variations:
        avg_learning_curve,timesteps= average_over_repetitions(enable_experience_replay, enable_target_network,n_eval_episodes,eval_interval, max_epoch_env_steps,n_repetitions, num_epochs, lr, discount_factor, mode, eps_start, smoothing_window)
        with open('evaluation_results_len.txt', 'a') as file:
            file.write(f'Evaluation Returns: {len(avg_learning_curve)}, Evaluation Timesteps: {len(timesteps)}\n')
        Plot.add_curve(timesteps, avg_learning_curve, label=label)
    
    Plot.save('dqn_variations.png')
    

if __name__ == '__main__':
    experiment()
