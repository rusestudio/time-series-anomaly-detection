"""
utils/lstm_model.py
LSTM Autoencoder architecture — must match HVAC_LSTM_Autoencoder.ipynb exactly.
"""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features=5, hidden_dim=64, bottleneck_dim=32,
                 n_layers=2, dropout=0.2):
        super().__init__()
        self.n_features     = n_features
        self.hidden_dim     = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers       = n_layers
        self.seq_len        = 96

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_dim,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0,
        )
        self.encoder_fc = nn.Linear(hidden_dim, bottleneck_dim)

        # Decoder
        self.decoder_fc = nn.Linear(bottleneck_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0,
        )
        self.output_fc = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        bottleneck  = self.encoder_fc(hidden[-1])
        dec_input   = self.decoder_fc(bottleneck).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_output, _ = self.decoder_lstm(dec_input)
        return self.output_fc(dec_output)

    def reconstruction_error(self, x_tensor):
        """Return per-window MSE error. x_tensor: (n, 96, 5)."""
        with torch.no_grad():
            recon = self.forward(x_tensor)
            mse   = ((recon - x_tensor) ** 2).mean(dim=(1, 2))
        return mse.cpu().numpy()
