import torch
import torch.nn as nn
import torch.nn.functional as F


class GINConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if adj.is_sparse:
            neigh = torch.sparse.mm(adj, x)
        else:
            neigh = adj @ x
        return F.relu(self.linear(x + neigh))


class TGAEEncoder(nn.Module):
    """T-GAE encoder used by the original graph-matching implementation."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: list[int],
                 output_dim: int,
                 num_hidden_layers: int):
        super().__init__()
        if len(hidden_dim) != num_hidden_layers + 1:
            raise ValueError(
                "hidden_dim must contain num_hidden_layers + 1 entries "
                f"(got {len(hidden_dim)} for {num_hidden_layers} layers)"
            )

        self.in_proj = nn.Linear(input_dim, hidden_dim[0])
        self.convs = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.convs.append(GINConv(input_dim + hidden_dim[i], hidden_dim[i + 1]))
        self.out_proj = nn.Linear(sum(hidden_dim), output_dim)

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        initial_x = x
        x = self.in_proj(x)
        hidden_states = [x]
        for layer in self.convs:
            x = layer(adj, torch.cat([initial_x, x], dim=1))
            hidden_states.append(x)
        return self.out_proj(torch.cat(hidden_states, dim=1))


class TGAENetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int | list[int] = 16,
                 output_dim: int = 8,
                 num_hidden_layers: int = 8):
        super().__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * (num_hidden_layers + 1)
        self.encoder = TGAEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_hidden_layers=num_hidden_layers,
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.encoder(adj, x)
