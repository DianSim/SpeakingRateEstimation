import torch
from torch import nn

class LSTMRegression(nn.Module):
    def __init__(self, input_size=560, hidden_size=128, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=1,
                            batch_first=True)
        self.hidden2out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # print('Input: ', x.shape)
        lstm_out, (hn, cn) = self.lstm(x)
        # print('LSTM out: ', lstm_out.shape)
        reg_out = self.hidden2out(lstm_out[:, -1, :])
        # print('reg_out: ', reg_out.shape)
        out = torch.clamp(reg_out, min=0, max=24)
        # print(out)
        return out