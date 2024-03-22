import torch
from torch import nn

class LSTMClassification(nn.Module):
    """Simple LSTM-RNN classification model to predict speaking rate 
        
        Args:
            input_size: input size of RNN-LSTM
            hidden_size: hidden size of RNN-LSTM
            output_size: output size of RNN-LSTM
    """
    def __init__(self, input_size=560, hidden_size=128, num_classes=25):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=1,
                            batch_first=True)
        self.hidden2out = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # print('Input: ', x.shape)
        lstm_out, (hn, cn) = self.lstm(x)
        # print('LSTM out: ', lstm_out.shape)
        # print(lstm_out[:, -1, :].shape)
        out = self.hidden2out(lstm_out[:, -1, :])
        # out = self.softmax(lin_out)
        # print('reg_out: ', out.shape)
        return out