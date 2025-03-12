import torch
import torch.nn as nn

class LayoutPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 64),  # Input: room dimensions + metadata
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 12)  # Output: x, y positions for 6 furniture items
        )
        
    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # Output normalized to [0, 1]

