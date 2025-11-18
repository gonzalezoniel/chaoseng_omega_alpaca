import torch
import torch.nn as nn

class OmegaModel(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, layers: int = 3):
        super().__init__()
        self.embed = nn.Linear(5, d_model)  # OHLCV -> embedding
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fc = nn.Linear(d_model, 3)  # buy, hold, sell

    def forward(self, x):
        # x: [seq_len, batch, features]
        x = self.embed(x)
        x = self.encoder(x)
        logits = self.fc(x[-1])  # last time step
        return logits
