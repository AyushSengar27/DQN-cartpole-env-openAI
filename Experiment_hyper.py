#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
from myDQNagent2 import DQN_learning
from Helper import LearningCurvePlot, smooth

def average_over_repetitions(enable_experience_replay, enable_target_network,n_eval_episodes,eval_interval, max_epoch_env_steps,n_repetitions, num_epochs, lr, discount_factor, mode, eps_start, temperature, smoothing_window, batch_size, hidden_units):
    returns_over_repetitions = []
    now = time.time()
    for repetition in range(n_repetitions):
        
        print("Repetition Number: ", repetition + 1)
        learning_curve,timesteps = DQN_learning(num_epochs=num_epochs,max_epoch_env_steps=max_epoch_env_steps,eval_interval=eval_interval,n_eval_episodes=n_eval_episodes,
                                      lr=lr,discount_factor=discount_factor,eps_start=eps_start,temperature=temperature,
                                      enable_target_network=enable_target_network,enable_experience_replay=enable_experience_replay,mode=mode,
                                      batch_size=batch_size,hidden_units=hidden_units)
        returns_over_repetitions.append(learning_curve)
    
    # Average the learning curves across repetitions
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    learning_curve = np.mean(np.array(returns_over_repetitions),axis=0) # average over repetitions  
    if smoothing_window is not None: 
        learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve, timesteps  

def experiment():
    print("Running experiments with hyperparameter tuning...")
    n_repetitions = 3
    num_epochs = 150
    max_epoch_env_steps=200
    eval_interval=5
    n_eval_episodes=10
    lr=0.001
    discount_factor=0.95
    eps_start=0.01
    temperature=0.1
    smoothing_window = 3
    plot = False
    enable_experience_replay=True
    enable_target_network=True
    batch_size=32
    hidden_units=[128,128]

    # Plot 1: different gamma values
    #gamma_values = [0.9, 0.95]
    gamma_values = [0.9, 0.95, 0.99, 1.0]
    mode = 'epsilon-greedy'
    Plot = LearningCurvePlot(title="DQN: effect of different gamma values")
    Plot.set_ylim(0, 250)
    for discount_factor in gamma_values:
        avg_learning_curve,timesteps= average_over_repetitions(enable_experience_replay, enable_target_network,n_eval_episodes,eval_interval, max_epoch_env_steps,n_repetitions, num_epochs, lr, discount_factor, mode, eps_start, temperature, smoothing_window, batch_size, hidden_units)
        Plot.add_curve(timesteps, avg_learning_curve, label=r'$\gamma $ = {}'.format(discount_factor))     
    Plot.save(name="DQN_gamma_performance.png")
    # plt.show()

    # Plot 2: different lr values
    lr_values = [0.0001, 0.001, 0.01, 0.1]
    mode = 'epsilon-greedy'
    Plot = LearningCurvePlot(title="DQN: effect of different learning rate values")
    Plot.set_ylim(0, 250)
    for lr in lr_values:
        avg_learning_curve,timesteps= average_over_repetitions(enable_experience_replay, enable_target_network,n_eval_episodes,eval_interval, max_epoch_env_steps,n_repetitions, num_epochs, lr, discount_factor, mode, eps_start, temperature, smoothing_window, batch_size, hidden_units)
        Plot.add_curve(timesteps, avg_learning_curve, label=r'learning rate = {}'.format(lr))     
    Plot.save(name="DQN_lr_performance.png")
    

    # # Plot 3: epsilon-greedy vs. softmax exploration
    mode = 'epsilon-greedy'
    epsilons = [0.01, 0.05, 0.2]
    Plot = LearningCurvePlot(title = 'Exploration: $\epsilon$-greedy versus softmax exploration')    
    Plot.set_ylim(0, 250)
    for eps_start in epsilons:        
        avg_learning_curve,timesteps= average_over_repetitions(enable_experience_replay, enable_target_network,n_eval_episodes,eval_interval, max_epoch_env_steps,n_repetitions, num_epochs, lr, discount_factor, mode, eps_start, temperature, smoothing_window, batch_size, hidden_units)
        with open('evaluation_results_egreedy.txt', 'a') as file:
            file.write(f'Evaluation Returns: {len(avg_learning_curve)}, Evaluation Timesteps: {len(timesteps)}\n')
        Plot.add_curve(timesteps, avg_learning_curve, label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(eps_start))    
    mode = 'softmax'
    temps = [0.01, 0.1, 1.0]
    for temperature in temps:
        avg_learning_curve,timesteps= average_over_repetitions(enable_experience_replay, enable_target_network,n_eval_episodes,eval_interval, max_epoch_env_steps,n_repetitions, num_epochs, lr, discount_factor, mode, eps_start, temperature, smoothing_window, batch_size, hidden_units)
        with open('evaluation_results_softmax.txt', 'a') as file:
            file.write(f'Evaluation Returns: {len(avg_learning_curve)}, Evaluation Timesteps: {len(timesteps)}\n')
        Plot.add_curve(timesteps, avg_learning_curve, label=r'softmax, $ \tau $ = {}'.format(temperature))
    mode = 'novelty-based'
    avg_learning_curve, timesteps = average_over_repetitions(enable_experience_replay, enable_target_network, n_eval_episodes, eval_interval, max_epoch_env_steps, n_repetitions, num_epochs, lr, discount_factor, mode, eps_start, temperature, smoothing_window, batch_size, hidden_units)
    Plot.add_curve(timesteps, avg_learning_curve, label='novelty-based')
    Plot.save('exploration.png')
    

    # Plot 4: different batch sizes [32, 64, 128]
    batch_size_values = [32, 64, 128]
    mode = 'epsilon-greedy'
    Plot = LearningCurvePlot(title="DQN: effect of different batch sizes")
    Plot.set_ylim(0, 250)
    for batch_size in batch_size_values:
        avg_learning_curve,timesteps= average_over_repetitions(enable_experience_replay, enable_target_network,n_eval_episodes,eval_interval, max_epoch_env_steps,n_repetitions, num_epochs, lr, discount_factor, mode, eps_start, temperature, smoothing_window, batch_size, hidden_units)
        Plot.add_curve(timesteps, avg_learning_curve, label=r'Batch size = {}'.format(batch_size))     
    Plot.save(name="DQN_batch_size.png")


    # # Plot 5: different number of layers and neurons
    hidden_units_configs = [
        [128, 128],  # 2 layers, 128 neurons each
        [64, 64, 64],  # 3 layers, 64 neurons each
        [256, 256],  # 2 layers, 256 neurons each
        [128, 128, 128],  # 3 layers, 128 neurons each
    ]
    mode = 'epsilon-greedy'
    Plot = LearningCurvePlot(title="DQN: Comparison of Network Architectures")
    Plot.set_ylim(0, 250)
    for hidden_units in hidden_units_configs:
        avg_learning_curve,timesteps= average_over_repetitions(enable_experience_replay, enable_target_network,n_eval_episodes,eval_interval, max_epoch_env_steps,n_repetitions, num_epochs, lr, discount_factor, mode, eps_start, temperature, smoothing_window, batch_size, hidden_units)
        label = f"{len(hidden_units)} layers, {hidden_units[0]} neurons"
        Plot.add_curve(timesteps, avg_learning_curve, label=label)
    Plot.save(name="DQN_network_architecture_comparison.png")
    

def dqn(enable_experience_replay, enable_target_network):
    print(f"Running DQN with ER={enable_experience_replay}, TN={enable_target_network}")
    
    n_repetitions = 3
    num_epochs = 150
    max_epoch_env_steps=200
    eval_interval=5
    n_eval_episodes=10
    lr=0.001
    discount_factor=0.95
    hidden_units=[128, 128]
    batch_size=32
    eps_start=0.01
    temperature=0.1
    mode='epsilon-greedy'
    smoothing_window = 9
    plot = False
    
    Plot = LearningCurvePlot(title='DQN Variation')    
    Plot.set_ylim(0, 300)
    
    if enable_experience_replay and enable_target_network:
        label = 'DQN with ER and TN'
    elif enable_experience_replay and not enable_target_network:
        label = 'DQN with ER'
    elif not enable_experience_replay and enable_target_network:
        label = 'DQN with TN'
    else:
        label = 'DQN without ER and TN'
    
    avg_learning_curve, timesteps = average_over_repetitions(enable_experience_replay, enable_target_network, n_eval_episodes, eval_interval, max_epoch_env_steps, n_repetitions, num_epochs, lr, discount_factor, mode, eps_start, temperature, smoothing_window, batch_size=batch_size,hidden_units=hidden_units)
    Plot.add_curve(timesteps, avg_learning_curve, label=label)
    
    plot_filename = f'{label.replace(" ", "_").lower()}.png'
    Plot.save(plot_filename)
   


def parse_args():
    parser = argparse.ArgumentParser(description='Run DQN experiments with optional configurations.')
    parser.add_argument('--dqn', action='store_true', help='DQN')
    parser.add_argument('--tn', action='store_true', help='Enable Target Network')
    parser.add_argument('--er', action='store_true', help='Enable Experience Replay')
    parser.add_argument('--experiments', action='store_true', help='Run experiments with hyperparameter tuning')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    

    if args.experiments:
        experiment()
    elif args.dqn:
        dqn(enable_experience_replay=args.er, enable_target_network=args.tn)
    else:
        print("Error: Please specify at least '--experiments' to run experiments with hyperparameter tuning, '--dqn' to run an experiment, or '--ablation' to run an ablation study.")