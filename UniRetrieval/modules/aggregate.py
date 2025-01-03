import torch
from typing import Union

from .layer import ActivationUnit


__all__ = [
    "AverageAggregator",
    "LastItemAggregator",
    "SelfAttentiveAggregator",
    "DeepInterestAggregator"
]

class AverageAggregator(torch.nn.Module):
    def __init__(self, dim=1, *args, **kwargs):
        """ Average sequence aggregator.

        Args:
            dim (int, optional): Dimension to average over. Defaults to 1.
        """
        super().__init__()
        self.dim = dim

    def forward(self, seq: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            seq (torch.Tensor): [batch_size, length, hidden_dim]
        """
        return torch.mean(seq, dim=self.dim)
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class LastItemAggregator(torch.nn.Module):
    def __init__(self, dim=1, *args, **kwargs):
        """ Get the representation of the last item as the representation of the sequence.

        Args:
            dim (int, optional): Dimension to aggregate. Defaults to 1.
        """
        super().__init__()
        self.dim = dim

    def forward(self, seq: torch.Tensor, padding_mask: torch.Tensor,  *args, **kwargs) -> torch.Tensor:
        """
        Args:
            seq (torch.Tensor): [batch_size, length, hidden_dim]
            padding_mask (torch.Tensor): [batch_size, length], padding masks

        Returns:
            torch.Tensor: [batch_size, hidden_dim]
        """
        lengths = (~padding_mask).sum(dim=-1)
        lengths = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, seq.size(-1))
        # in case of -1, replace it with 0
        lengths = lengths.clamp(min=0)
        last_item = seq.gather(dim=self.dim, index=lengths)  # [batch_size, 1, hidden_dim]
        return last_item.squeeze(dim=self.dim)  # [batch_size, hidden_dim]

    
    def extra_repr(self) -> str:
        return f"dim={self.dim}"
    


class SelfAttentiveAggregator(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            max_seq_len: int,
            n_layers: int=1,
            n_heads: int=4,
            hidden_size: int=512,
            dropout: float=0.3,
            activation: Union[str, torch.nn.Module]='relu',
            *args,
            **kwargs
        ):
        super().__init__()
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation
        encoder_layers = torch.nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=hidden_size,
            batch_first=True,
            activation=activation,
            dropout=dropout,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers=self.n_layers)
        self.position_embedding = torch.nn.Embedding(max_seq_len, input_dim)
        self.seq_aggragation = LastItemAggregator(dim=1)
    

    def forward(self, seq: torch.Tensor, padding_mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        device = padding_mask.device
        position_ids = torch.arange(seq.shape[1], dtype=torch.long).unsqueeze(0).repeat(seq.shape[0], 1).to(device)    # BxL
        seq_pos_emb = self.position_embedding(position_ids)   # BxLxD
        # count nonzero items in ~padding_mask
        seq_emb_with_pos = seq + seq_pos_emb
        seq_emb = self.transformer_encoder(seq_emb_with_pos, src_key_padding_mask=padding_mask)
        last_emb = self.seq_aggragation.forward(seq_emb, padding_mask) # BxD
        return last_emb
    

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, n_layers={self.n_layers}, n_heads={self.n_heads}, hidden_dim={self.hidden_size}, dropout={self.dropout}"
    


class DeepInterestAggregator(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_size: int=512,
            *args,
            **kwargs
        ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.activation_unit = ActivationUnit(input_dim, hidden_size)
    
    def forward(self, seq: torch.Tensor, padding_mask: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            seq (torch.Tensor): [batch_size, seq_length, hidden_dim], sequence of embeddings
            padding_mask (torch.Tensor): [batch_size, seq_length], mask for padding
            target (torch.Tensor): [batch_size, hidden_dim], target embedding
        """
        weights = self.activation_unit(query=target, keys=seq)  # [B, T, 1]
        weights[padding_mask] = 0.0
        return (seq * weights).sum(dim=1)   # [B, D]