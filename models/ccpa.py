import torch
import torch.nn as nn


class ClimateConditioner(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class PhenologicalAdapter(nn.Module):
    def __init__(self, feature_dim=64, condition_dim=32):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )
        self.film_scale = nn.Linear(condition_dim, feature_dim)
        self.film_shift = nn.Linear(condition_dim, feature_dim)
        self.output = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )

    def forward(self, features, climate_condition):
        h = self.feature_encoder(features)
        scale = self.film_scale(climate_condition)
        shift = self.film_shift(climate_condition)
        h = scale * h + shift
        return self.output(h)


class CCPA(nn.Module):
    def __init__(self, input_dim=5, condition_dim=32, feature_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
        )
        self.conditioner = ClimateConditioner(input_dim=input_dim, hidden_dim=64, output_dim=condition_dim)
        self.adapter = PhenologicalAdapter(feature_dim=feature_dim, condition_dim=condition_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        climate_vec = self.conditioner(x)
        adapted = self.adapter(features, climate_vec)
        return self.head(adapted).squeeze()


class CCPANoClimate(nn.Module):
    """Ablation: CCPA without climate conditioning."""
    def __init__(self, input_dim=5, feature_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim, feature_dim), nn.ReLU())
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.ReLU(),
            nn.Linear(feature_dim, feature_dim), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(feature_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.adapter(self.backbone(x))).squeeze()
