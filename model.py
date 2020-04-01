import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_size=64, fc2_size=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_size (int): Number of nodes in first hidden layer
            fc2_size (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class QNetwork2(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_size=96, fc2_size=96):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_size (int): Number of nodes in first hidden layer
            fc2_size (int): Number of nodes in second hidden layer
            fc3_size (int): Number of nodes in second hidden layer
        """
        super(QNetwork2, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.drop_layer1 = nn.Dropout(0.35)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.drop_layer2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(fc2_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = self.drop_layer1(x)
        x = F.relu(self.fc2(x))
        x = self.drop_layer2(x)
        x = self.fc3(x)
        return x
