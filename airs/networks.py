# airs/networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """
    Shared conv encoder used by the policy/value network.
    """
    def __init__(self, obs_shape, feature_dim=256):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute conv output size by doing a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_dim = self.conv(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, feature_dim),
            nn.ReLU()
        )
        self.feature_dim = feature_dim

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns: [B, feature_dim]
        """
        z = self.conv(x)
        z = self.fc(z)
        return z


class ActorCriticNet(nn.Module):
    """
    Simple A2C network for MiniGrid:
    - Conv encoder
    - Policy head
    - Value head
    """
    def __init__(self, obs_shape, num_actions, feature_dim=256):
        super().__init__()
        self.encoder = ConvEncoder(obs_shape, feature_dim=feature_dim)
        self.policy = nn.Linear(feature_dim, num_actions)
        self.value = nn.Linear(feature_dim, 1)

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns: logits [B, A], value [B]
        """
        z = self.encoder(x)
        logits = self.policy(z)
        value = self.value(z).squeeze(-1)
        return logits, value


class RandomEncoder(nn.Module):
    """
    Random, fixed encoder for RE3.
    We freeze parameters after initialization.
    """
    def __init__(self, obs_shape, feature_dim=128):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_dim = self.conv(dummy).shape[1]
        self.fc = nn.Linear(conv_out_dim, feature_dim)
        self.feature_dim = feature_dim

        # Initialize randomly and freeze
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        """
        x: [B, C, H, W]
        returns: [B, feature_dim]
        """
        z = self.conv(x)
        z = self.fc(z)
        return z
