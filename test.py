import gym
import torch
from model import QNetwork
from utils import *

episodes = 10000
    
def select_action(state):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        return q_net(state).argmax().item()

env = gym.make('Blackjack-v1')

state_dim = 3
action_dim = env.action_space.n

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

q_net = QNetwork(state_dim, action_dim).to(device)
q_net.load_state_dict(torch.load('best_black_jack.pth'))
q_net.eval()

for i in range(10):
    win = 0
    draw = 0
    lose = 0
    for episode in range(1, episodes + 1):
        state = env.reset()[0]
        state = encode_state(state)
        done = False

        while not done:
            action = select_action(state)
            next_state, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = encode_state(next_state)
            state = next_state

            if done:
                if env_reward == 1:
                    win += 1
                elif env_reward == 0:
                    draw += 1
                else:
                    lose += 1

    print(f"Win: {win / episodes:.2f}, Draw: {draw / episodes:.2f}, Lose: {lose / episodes:.2f}")