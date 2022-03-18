import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn


class AutoRec(nn.Module):

    def __init__(self,
                input_size: int,
                hidden_size: int):
        super(AutoRec, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_layer = nn.Linear(in_features=self.input_size,
                                out_features=self.hidden_size,
                                bias=True)
        self.output_layer = nn.Linear(in_features=self.hidden_size,
                                 out_features=self.input_size,
                                 bias=True)


    def forward(self, x):
        hidden = self.input_layer(x)
        output = self.output_layer(hidden)
        return output


def main():
    batch_size = 16
    input_size=50
    hidden_size=200

    autorec = AutoRec(input_size=input_size,
                      hidden_size=hidden_size)

    dummy_data = np.array([[0 for _ in range(input_size)] for _ in range(batch_size)])
    dummy_data = torch.from_numpy(dummy_data).float()
    result = autorec(dummy_data)

    # torch.Size([16, 50])
    # same with batch_size, input_size
    print(result.shape) 


if __name__ == "__main__":
    main()