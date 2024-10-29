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
            nn.Linear(state_dim, 32),
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
        q_values = []
        for action, values in zip(actions, action_values):
            q = values.max() if target else values[action]
            q_values.append(q)
        
        # x = compute_global_relative_features(x, num_agents)
        x = self.fcin(x)
        agent_features = x[:num_agents]
        
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
        
class Qatten(nn.Module):
    """
    Qatten model with attention mechanism.
    """
    def __init__(self, state_dim=26, num_heads=8):
        super(Qatten, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads)
        
        self.fcin = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        self.initialize_weights()

    def forward(self, x, actions, action_values, num_agents=3, target=False):
        """
        Forward pass through the Graph Qatten model.
        """
        q_values = []
        for action, values in zip(actions, action_values):
            q = values.max() if target else values[action]
            q_values.append(q)

        x = self.fcin(x)
        agent_features = x[:num_agents]

        _, weights = self.attention(query=agent_features, key=agent_features, value=agent_features)
        weights = weights.sum(dim=0).unsqueeze(0)
        weights = F.softmax(weights, dim=1)
        q_values = torch.stack(q_values).unsqueeze(1)
        q_tot = torch.matmul(weights, q_values)
        
        return q_tot.squeeze(1)
    
    def initialize_weights(self):
        self.apply(initialize_weights)

class AIQatten(nn.Module):
    """
    Graph Qatten model with GAT layers and attention mechanism.
    """
    def __init__(self, state_dim=26, value_dim=5, num_heads=8):
        super(AIQatten, self).__init__()
        self.fcstate = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        self.fcvalue = nn.Sequential(
            nn.Linear(value_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        self.attention = nn.MultiheadAttention(num_heads=num_heads, embed_dim=128)

        self.initialize_weights()

    def forward(self, x, actions, action_values, num_agents=3, target=False):
        """
        Forward pass through the Graph Qatten model.
        """
        # Process actions and action values
        q_values = []
        for action, values in zip(actions, action_values):
            q = values.max() if target else values[action]
            q_values.append(q)

        action_values = torch.stack(action_values)

        # Pass node features through the state linear layers
        x = self.fcstate(x)
        action_values = self.fcvalue(action_values)

        agent_features = torch.cat([x[:num_agents], action_values], dim=1)

        # Apply value attention mechanism
        _, weights = self.attention(query=agent_features, key=agent_features, value=agent_features)
        
        # Sum attention weights and apply softmax
        weights = weights.sum(dim=0).unsqueeze(0)
        weights = F.softmax(weights, dim=1)

        # Stack Q-values and apply weighted sum
        q_values = torch.stack(q_values).unsqueeze(1)
        q_tot = torch.matmul(weights, q_values)
        
        return q_tot.squeeze(1)

    def initialize_weights(self):
        self.apply(initialize_weights)