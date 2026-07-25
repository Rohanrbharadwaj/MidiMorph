"""
MusicVAE architecture: bidirectional LSTM encoder, z-conditioned categorical
LSTM decoder, and the VAE wrapper with the reconstruction + KL loss.

Defaults match the shapes the checkpoint on Drive was trained with
(OUTPUT_DEPTH=90, ENC_RNN_SIZE=512, dec_hidden=256, z_size=256). If you
change any of these when loading a checkpoint, the state_dict won't match.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import OUTPUT_DEPTH

ENC_RNN_SIZE = 512  # hidden size of each direction's LSTM


class BidirectionalLstmEncoder(nn.Module):
    def __init__(self, input_size: int = OUTPUT_DEPTH, hidden_size: int = ENC_RNN_SIZE):
        super().__init__()
        # LSTM (not LSTMCell) for faster GPU training via cuDNN's fused kernel
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, input_size] -> [batch, 2*hidden_size]"""
        _, (h_n, c_n) = self.lstm(x)
        h_fw, h_bw = h_n[0], h_n[1]
        return torch.cat([h_fw, h_bw], dim=-1)


class CategoricalLstmDecoder(nn.Module):
    def __init__(
        self,
        output_depth: int = OUTPUT_DEPTH,
        hidden_size: int = 256,
        z_size: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.output_depth = output_depth
        self.hidden_size = hidden_size
        self.z_size = z_size
        self.num_layers = num_layers

        dec_input_size = output_depth + z_size
        self.input_proj = nn.Linear(dec_input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.z_to_state = nn.Linear(z_size, num_layers * 2 * hidden_size)
        self.output_projection = nn.Linear(hidden_size, output_depth)

    def initial_state(self, z: torch.Tensor):
        state = torch.tanh(self.z_to_state(z))
        chunks = torch.chunk(state, self.num_layers * 2, dim=-1)
        c_list = chunks[0::2]
        h_list = chunks[1::2]
        h0 = torch.stack(h_list, dim=0)
        c0 = torch.stack(c_list, dim=0)
        return (h0, c0)

    def forward_teacher_forced(self, x_input: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x_input.shape
        z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)
        dec_in = torch.cat([x_input, z_expanded], dim=-1)
        dec_in = self.input_proj(dec_in)

        h0, c0 = self.initial_state(z)
        output, _ = self.lstm(dec_in, (h0, c0))
        logits = self.output_projection(output)
        return logits

    @torch.no_grad()
    def sample(self, z: torch.Tensor, max_length: int = 32, temperature: float = 1.0) -> torch.Tensor:
        """Autoregressive sampling. Returns token indices [batch, max_length]."""
        batch_size = z.shape[0]
        device = z.device
        h, c = self.initial_state(z)
        prev_token = torch.zeros(batch_size, self.output_depth, device=device)

        samples = []
        for _ in range(max_length):
            dec_in = torch.cat([prev_token, z], dim=-1).unsqueeze(1)
            dec_in = self.input_proj(dec_in)
            out, (h, c) = self.lstm(dec_in, (h, c))
            logits = self.output_projection(out.squeeze(1))
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).squeeze(-1)
            prev_token = F.one_hot(idx, self.output_depth).float()
            samples.append(idx)
        return torch.stack(samples, dim=1)


class MusicVAE(nn.Module):
    def __init__(
        self,
        output_depth: int = OUTPUT_DEPTH,
        enc_hidden: int = ENC_RNN_SIZE,
        dec_hidden: int = 256,
        z_size: int = 256,
    ):
        super().__init__()
        self.encoder = BidirectionalLstmEncoder(output_depth, enc_hidden)
        self.decoder = CategoricalLstmDecoder(output_depth, dec_hidden, z_size)

        enc_output_size = 2 * enc_hidden
        self.fc_mu = nn.Linear(enc_output_size, z_size)
        self.fc_sigma = nn.Linear(enc_output_size, z_size)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        sigma = F.softplus(self.fc_sigma(h))
        return mu, sigma

    def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x: torch.Tensor):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        x_input = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)
        logits = self.decoder.forward_teacher_forced(x_input, z)
        return logits, mu, sigma

    def loss(self, x: torch.Tensor):
        logits, mu, sigma = self.forward(x)
        target_idx = x.argmax(dim=-1)
        r_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target_idx.reshape(-1), reduction="none"
        ).reshape(x.shape[0], x.shape[1])
        r_loss = r_loss.sum(dim=-1)
        kl = 0.5 * (mu**2 + sigma**2 - torch.log(sigma**2) - 1).sum(dim=-1)
        return r_loss, kl
