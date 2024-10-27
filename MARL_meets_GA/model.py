import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) with Graph Attention Network (GAT) layers and GRU integration.
    """
    def __init__(self, state_dim=26, out_dim=4):
        super(DQN, self).__init__()        
        self.fcin = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        self.fcout = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, out_dim)
        )
        
        self.initialize_weights()

    def forward(self, x, id):
        """
        Forward pass through the DQN model with GAT and linear layers.
        """
        x = self.fcin(x[id])
        out = self.fcout(x)
        return out

    def initialize_weights(self):
        self.apply(initialize_weights)

class QMIX(nn.Module):
    """
    QMIX network that mixes individual agent Q-values into a global Q-value.
    """
    def __init__(self, state_dim=26):
        super(QMIX, self).__init__()
        # Linear layers managed by Sequential
        self.fcin = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        # Mixing network components
        self.w1 = nn.Linear(64, 64)
        self.b1 = nn.Linear(64, 64)
        self.w2 = nn.Linear(64, 1)
        self.b2 = nn.Linear(64, 1)
        
        self.initialize_weights()

    def forward(self, x, actions, action_values, num_agents=3, target=False):
        actions = [int(action) for agent, action in actions.items() if 'predator' in agent]
        q_values = []
        for action, values in zip(actions, action_values):
            q = values.max() if target else values[action]
            q_values.append(q)
        
        # node_features = compute_global_relative_features(node_features, num_agents)
        node_features = self.fcin(node_features)
        agent_features = node_features[:num_agents]
        
        # Stack Q-values
        q_values = torch.stack(q_values).unsqueeze(-1)  # Make sure q_values has a feature dimension
        
        # Apply the mixing network
        w1 = F.relu(self.w1(agent_features))  # Weighting Q-values by agent features
        b1 = self.b1(agent_features)  # Bias term
        hidden_qs = q_values * w1 + b1  # Mixing Q-values
        
        w2 = F.relu(self.w2(agent_features))  # Second weight layer
        b2 = self.b2(agent_features)  # Second bias layer
        q_tot = torch.matmul(hidden_qs.transpose(-1, -2), w2) + torch.mean(b2, dim=0)  # Global Q-value calculation
        
        return q_tot.sum()

    def initialize_weights(self):
        self.apply(initialize_weights)