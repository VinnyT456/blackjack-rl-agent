import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LayerNorm(8),
            nn.LeakyReLU(),
            nn.Linear(8, action_dim),
        )

    def forward(self, x):
        return self.fc(x)