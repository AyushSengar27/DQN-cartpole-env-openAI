import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import gym
import random
from collections import deque
import warnings
import os
from dataclasses import dataclass, field

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

@dataclass
class SimpleReplayBuffer:
    capacity: int
    buffer: deque = field(init=False, default_factory=lambda: deque(maxlen=10000))

    def update_buffer(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        samples = random.sample(self.buffer, sample_size)
        s, a, r, next_s, done = zip(*samples)
        return np.array(s), np.array(a), np.array(r), np.array(next_s), np.array(done)

    @property
    def size(self):
        return len(self.buffer)
    
@dataclass
class DQNAgent:
    state_dimensions: any
    action_space: any
    lr: float = 0.001
    discount_factor: float = 0.8
    eps_start: float = 0.1  # Epsilon value for epsilon-greedy
    temperature: float = 1.0 # Temperature value for softmax
    hidden_units: list = field(default_factory=lambda: [64, 64])
    enable_target_network: bool = False
    enable_experience_replay: bool = True
    replay_buffer_size: int = 1000
    batch_size: int = 32
    online_network: keras.Sequential = field(init=False)
    target_network: keras.Sequential = field(init=False, default=None)
    replay_buffer: SimpleReplayBuffer = field(init=False)
    num_actions: int = field(init=False)

    def __post_init__(self):
        self.state_dimensions = self.state_dimensions.shape
        self.num_actions = self.action_space.n
        self.online_network = self.build_network()
        self.target_network = self.build_network() if self.enable_target_network else None
        self.replay_buffer = SimpleReplayBuffer(self.replay_buffer_size)
        self.visit_counts = {}

    def build_network(self):
        model = keras.Sequential()
        model.add(layers.Input(shape=self.state_dimensions))
        for n_nodes in self.hidden_units:
            model.add(layers.Dense(n_nodes, activation='relu', kernel_initializer='HeUniform'))
        model.add(layers.Dense(self.num_actions, activation='linear', kernel_initializer='HeUniform'))
        model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=self.lr))
        return model

    def softmax_action_selection(self, state, temperature=1.0):
        q_values = self.online_network.predict(state.reshape(1, -1)).flatten()
        scaled_q_values = q_values / temperature
        max_q = np.max(scaled_q_values)
        exp_q_values = np.exp(scaled_q_values - max_q)
        probabilities = exp_q_values / np.sum(exp_q_values)
        action = np.random.choice(self.num_actions, p=probabilities)
        return action

    def calculate_novelty_score(self, state, action):
        state_action_key = (tuple(state), action)
        self.visit_counts[state_action_key] = self.visit_counts.get(state_action_key, 0) + 1
        novelty_score = 1 / self.visit_counts[state_action_key]
        return novelty_score
    
    def select_action(self, state, mode='epsilon-greedy'):
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if mode == 'greedy':
            q_values = self.online_network.predict(state.reshape(1, -1))
            return np.argmax(q_values[0])
        elif mode == 'epsilon-greedy':
            if np.random.rand() <= self.eps_start:
                return random.randrange(self.num_actions)
            else:
                q_values = self.online_network.predict(state.reshape(1, -1))
                return np.argmax(q_values[0])
        elif mode == 'softmax':
            return self.softmax_action_selection(state, self.temperature)
        # elif mode == 'softmax':
        #     q_values = self.online_network.predict(state.reshape(1, -1)).flatten()
        #     probabilities = np.exp(q_values) / np.sum(np.exp(q_values))
        #     return np.random.choice(self.num_actions, p=probabilities)
        elif mode == 'novelty-based':
            novelty_scores = np.array([self.calculate_novelty_score(state, a) for a in range(self.num_actions)])
            action_probabilities = novelty_scores / novelty_scores.sum()
            return np.random.choice(self.num_actions, p=action_probabilities)
        
        else:
            raise ValueError("Invalid mode for action selection")

    def evaluate(self, eval_env, n_eval_episodes=10, max_episode_length=200):
        returns = []  
        for _ in range(n_eval_episodes):
            s, _ = eval_env.reset()
            R_ep = 0
            for _ in range(max_episode_length):
                a = self.select_action(s, mode='greedy')
                s_prime, r, done, _,_= eval_env.step(a)
                R_ep += r
                s = s_prime
                if done:
                    break
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return

    def update_network_with_sample(self, s, a, r, next_s, done):
        target = self.online_network.predict(s)
        next_target = self.target_network.predict(next_s) if self.enable_target_network else self.online_network.predict(next_s)

        for i in range(len(s)):
            target[i, a[i]] = r[i] if done[i] else r[i] + self.discount_factor * np.max(next_target[i])

        self.online_network.fit(s, target, batch_size=len(s), verbose=0)

    def experience_replay(self):
        if not self.enable_experience_replay or self.replay_buffer.size < self.batch_size:
            return
        s, a, r, next_s, done = self.replay_buffer.sample(self.batch_size)
        target = self.online_network.predict(s)
        next_target = self.target_network.predict(next_s) if self.enable_target_network else self.online_network.predict(next_s)

        for i in range(self.batch_size):
            target[i, a[i]] = r[i] if done[i] else r[i] + self.discount_factor * np.max(next_target[i])

        self.online_network.fit(s, target, batch_size=self.batch_size, verbose=0)

    def synchronize_networks(self):
        if self.enable_target_network:
            self.target_network.set_weights(self.online_network.get_weights())

def run():
    num_epochs = 150
    max_epoch_env_steps = 300
    eval_interval = 10  # Number of epochs between evaluations
    n_eval_episodes = 5  # Number of episodes to run for evaluation

    env = gym.make('CartPole-v1')
    eval_env = gym.make('CartPole-v1')  # Separate environment for evaluation
    agent = DQNAgent(env.observation_space, env.action_space, lr=0.001, discount_factor=0.95, eps_start=0.01, temperature=0.1, hidden_units=[128, 128], enable_target_network=False, enable_experience_replay=True, replay_buffer_size=10000, batch_size=32)

    eval_results_file = 'eval_results.txt'
    if os.path.exists(eval_results_file):
        os.remove(eval_results_file)  # Clear the eval results file at the start

    for epoch in range(num_epochs):
        s,_ = env.reset()
        total_reward = 0
        for timestep in range(max_epoch_env_steps):
            a = agent.select_action(s)
            next_s, reward, done, _,_ = env.step(a)
            if agent.enable_experience_replay:
                agent.replay_buffer.update_buffer((s, a, reward, next_s, done))
            else:
                agent.update_network_with_sample(np.array([s]), np.array([a]), np.array([reward]), np.array([next_s]), np.array([done]))
            s = next_s
            total_reward += reward
            if agent.enable_experience_replay :
                agent.experience_replay()
            if done or (timestep == max_epoch_env_steps-1):
                if agent.enable_target_network:
                    agent.synchronize_networks()
                break

        

        # Evaluation and logging
        if epoch % eval_interval == 0:
            avg_eval_reward = agent.evaluate(eval_env, n_eval_episodes)
            print(f"Epoch {epoch}: Avg Eval Reward = {avg_eval_reward}")
            # Append evaluation results to the file
            with open(eval_results_file, 'a') as eval_file:
                eval_file.write(f"Epoch: {epoch}, Avg Eval Reward: {avg_eval_reward}\n")

        # Append training results to the file in real-time
        with open('results.txt', 'a') as file:
            file.write(f"Epoch: {epoch}, Total reward: {total_reward}, Epsilon: {agent.eps_start:.2f}\n")



def DQN_learning(num_epochs,max_epoch_env_steps,eval_interval,n_eval_episodes,lr=0.001,discount_factor=0.95,eps_start=0.01,temperature=0.1,enable_target_network=False, enable_experience_replay=True,mode='epsilon-greedy',batch_size=32,hidden_units=[128, 128]):
        mode=mode
        num_epochs = num_epochs
        max_epoch_env_steps = max_epoch_env_steps
        eval_interval = eval_interval  # Number of epochs between evaluations
        n_eval_episodes = n_eval_episodes  # Number of episodes to run for evaluation
        eval_timesteps = []
        eval_returns = []

        env = gym.make('CartPole-v1')
        eval_env = gym.make('CartPole-v1')  # Separate environment for evaluation
        agent = DQNAgent(env.observation_space, env.action_space, lr=lr, discount_factor=discount_factor, eps_start=eps_start, temperature=temperature, hidden_units=[128, 128], enable_target_network=enable_target_network, enable_experience_replay=enable_experience_replay, replay_buffer_size=10000, batch_size=32)
        counter=0

        for epoch in range(num_epochs):
            s,_ = env.reset()
            total_reward = 0
            for timestep in range(max_epoch_env_steps):
                a = agent.select_action(s, mode=mode)
                next_s, reward, done, _,_ = env.step(a)
                counter+=1
                if agent.enable_experience_replay:
                    agent.replay_buffer.update_buffer((s, a, reward, next_s, done))
                else:
                    agent.update_network_with_sample(np.array([s]), np.array([a]), np.array([reward]), np.array([next_s]), np.array([done]))
                s = next_s
                total_reward += reward
                if agent.enable_experience_replay :
                    agent.experience_replay()
                if counter%10==0:
                        if agent.enable_target_network:
                            agent.synchronize_networks()

                if done or (timestep == max_epoch_env_steps-1):

                    break

            

            # Evaluation and logging
            if epoch % eval_interval == 0:
                avg_eval_reward = agent.evaluate(eval_env, n_eval_episodes)
                eval_returns.append(avg_eval_reward)
                eval_timesteps.append(epoch)

        return np.array(eval_returns), np.array(eval_timesteps)  



def test():
    
    num_epochs = 50
    max_epoch_env_steps = 50
    eval_interval = 5  # Number of epochs between evaluations
    n_eval_episodes = 10
    enable_target_network = False
    enable_experience_replay=True


    eval_returns, eval_timesteps = DQN_learning(num_epochs,max_epoch_env_steps,eval_interval,n_eval_episodes,lr=0.001,discount_factor=0.95,eps_start=0.01,temperature=0.1,enable_target_network=enable_target_network, enable_experience_replay=enable_experience_replay,mode='epsilon-greedy')
    print(len(eval_returns),len(eval_timesteps))
    with open('evaluation_results.txt', 'a') as file:
        file.write(f'Evaluation Returns: {eval_returns}, Evaluation Timesteps: {eval_timesteps}\n')

if __name__ == '__main__':
    test()


