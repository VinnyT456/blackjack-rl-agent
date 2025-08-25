import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from model import QNetwork
from utils import *

#Set the random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#Hyperparameters
gamma = 0.99
lr = 5e-4
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01
batch_size = 128
target_update = 10
memory_size = 10000
episodes = 20000

#Setup the environment and seeds.
env = gym.make('Blackjack-v1')

env.reset(seed=SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

#Dimension of the state and action
state_dim = 3
action_dim = env.action_space.n

#Initialize the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Initialize the neural networks
q_net = QNetwork(state_dim, action_dim).to(device)
target_net = QNetwork(state_dim, action_dim).to(device)
target_net.load_state_dict(q_net.state_dict())

#Initialize the optimizer
optimizer = optim.Adam(q_net.parameters(), lr=lr)

#Create the memory deque 
memory = deque(maxlen=memory_size)

#Change the mode of the neural networks
q_net.train()   
target_net.eval()

max_reward = -float('inf')

#Trains the model
def train_model():
    #Start training the neural network whenever the memory gets past the batch_size
    if len(memory) < batch_size:
        return 
    #Randomly choose a sample size of (batch_size) from the memory for training
    batch = random.sample(memory, batch_size)

    #Split it into state, action, reward, next state, and done
    s, a, r, s_, d = zip(*batch)

    #Turn it all into tensors and add an extra dimension if necessary
    s = torch.from_numpy(np.array(s)).float().to(device)
    a = torch.LongTensor(a).unsqueeze(1).to(device)
    r = torch.FloatTensor(r).unsqueeze(1).to(device)
    s_ = torch.from_numpy(np.array(s_)).float().to(device)
    d = torch.FloatTensor(d).unsqueeze(1).to(device)

    #Grab the q values in each row based on the index given by the actions
    q_pred = q_net(s).gather(1, a)

    #Get the action based on the max q value from each row using the next state
    online_next_actions = q_net(s_).argmax(dim=1, keepdim=True)

    #Grab the q values in each row based on the actions and detach it 
    next_q = target_net(s_).gather(1, online_next_actions).detach() 

    #Find the new q value with the formula
    q_target = r + gamma * next_q * (1 - d)

    #Compute loss, zero gradience, back propagation, step the optimizer
    loss = nn.MSELoss()(q_pred, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#For the first 500 episodes teach the agent to play by the custom action then random
def select_action(state, epsilon, episode):
    if (episode < 500):
        return custom_action(state)
    if random.random() < epsilon:
        return env.action_space.sample()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        return q_net(state).argmax().item()

#Training Loop
for episode in range(1,episodes+1):
    state = env.reset()[0]
    state = encode_state(state)

    total_reward = 0
    done = False

    while not done:
        #Select and play the action
        action = select_action(state, epsilon, episode)
        next_state, env_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = encode_state(next_state)

        #Give the agent a small bonus based on their action and bust chance
        shaped_reward = env_reward + 0.01 * custom_reward(state, action)

        #Add the state, action, reward, next state, and done or not to the memory deque
        memory.append((state, action, shaped_reward, next_state, done))

        #Change the state and add the reward
        state = next_state
        total_reward += shaped_reward
        train_model()

    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Update target network
    if episode % target_update == 0:
        target_net.load_state_dict(q_net.state_dict())

    #Save the model with the best performance
    if total_reward > max_reward:
        max_reward = total_reward
        torch.save(q_net.state_dict(), 'best_black_jack.pth')

    print(f"Episode: {episode}, Total Reward: {total_reward}")
