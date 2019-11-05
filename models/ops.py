import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor):
        x = x.contiguous()
        return x.view(x.size(0), -1)

class CNN_Permute(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.permute(0, 2, 1)

class RNN_Permute(nn.Module):
    def forward(self, x: torch.Tensor):
        x = x.permute(2,0,1)
        x = x.view(x.size(0), x.size(1), -1)
        return x