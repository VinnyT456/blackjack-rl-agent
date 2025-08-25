import gymnasium as gym
import torch
from model import QNetwork
from utils import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

episodes = 10000
SEED = 24

player_sum = range(12,22)
dealer_sum = range(1, 11)

wins = np.zeros((len(player_sum), len(dealer_sum)))
draws = np.zeros((len(player_sum), len(dealer_sum)))
loss = np.zeros((len(player_sum), len(dealer_sum)))
    
def select_action(state):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        return q_net(state).argmax().item()

env = gym.make('Blackjack-v1')
env.reset(seed=SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

state_dim = 3
action_dim = env.action_space.n

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

q_net = QNetwork(state_dim, action_dim).to(device)
q_net.load_state_dict(torch.load('best_black_jack.pth'))
q_net.eval()

average_win_rate = []

win = 0
draw = 0
lose = 0

hit = 0
stick = 0

for episode in range(1, episodes + 1):
    state = env.reset()[0]
    player_sum = state[0]-12
    dealer_sum = state[1]-1
    state = encode_state(state)
    done = False

    while not done:
        action = select_action(state)
        if (action == 0):
            hit+=1
        else:
            stick+=1
        next_state, env_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = encode_state(next_state)
        state = next_state

        if done:
            if env_reward == 1:
                win += 1
                wins[player_sum,dealer_sum]+=1
            elif env_reward == 0:
                draw += 1
                draws[player_sum,dealer_sum]+=1
            else:
                lose += 1
                loss[player_sum,dealer_sum]+=1

    if (episode % 100 == 0):
        average_win_rate.append(win / episode)

print(f"Win: {win / episodes:.2f}, Draw: {draw / episodes:.2f}, Lose: {lose / episodes:.2f}")

totals = wins + draws + loss
win_rate = np.divide(wins, totals, out=np.zeros_like(wins), where=totals!=0)
draw_rate = np.divide(draws, totals, out=np.zeros_like(wins), where=totals!=0)
loss_rate = np.divide(loss, totals, out=np.zeros_like(wins), where=totals!=0)

fig,axes = plt.subplots(2,3,figsize=(24, 8))

sns.heatmap(
    win_rate*100,
    xticklabels=range(1,11),
    yticklabels=range(12,22),
    ax=axes[0,0],
    cmap='Blues',
    annot=True,
    cbar=True,
    fmt=".0f"
)
axes[0,0].set_title("Win %")
axes[0,0].set_xlabel("Dealer Showing")
axes[0,0].set_ylabel("Player Sum")

sns.heatmap(
    draw_rate*100,
    xticklabels=range(1,11),
    yticklabels=range(12,22),
    ax=axes[0,1],
    cmap='Greens',
    annot=True,
    cbar=True,
    fmt=".0f"
)
axes[0,1].set_title("Draw %")
axes[0,1].set_xlabel("Dealer Showing")
axes[0,1].set_ylabel("Player Sum")

sns.heatmap(
    loss_rate*100,
    xticklabels=range(1,11),
    yticklabels=range(12,22),
    ax=axes[0,2],
    cmap='Reds',
    annot=True,
    cbar=True,
    fmt=".0f"
)
axes[0,2].set_title("Loss %")
axes[0,2].set_xlabel("Dealer Showing")
axes[0,2].set_ylabel("Player Sum")

data = [win,draw,lose]
axes[1,0].pie(data,labels=["Win","Draw","Lose"],autopct='%.0f%%')

axes[1,1].plot(average_win_rate)
axes[1,1].set_title("Average Win Rate Over Episodes")
axes[1,1].set_xlabel("Episodes (x100)") 
axes[1,1].set_ylabel("Win Rate")

sns.barplot(
    x=["Hit", "Stick"],     
    y=[hit, stick],   
    hue=["Hit","Stick"],  
    palette="Set1",    
    ax=axes[1,2]
)
axes[1,2].set_title("Hit vs. Stick")
axes[1,2].set_ylabel("Times Chosen")

plt.subplots_adjust(
    top=0.95,    
    bottom=0.1,
    hspace=0.4,
    wspace=0.3   
)

plt.show()
