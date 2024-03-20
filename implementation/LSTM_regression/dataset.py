"""Dataset class for LSTM regression model"""
import sys
import os
from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
import glob
from config import config


class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.file_list = glob.glob(os.path.join(data_dir, '**/*.wav'), recursive=True)

    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        waveform = torch.squeeze(waveform)
        file_name = file_path.split(os.path.sep)[-1]
        label = file_name.split('.')[0].split('_')[0]
        label = torch.tensor(int(label), dtype=torch.float32)
        return waveform, label


def collate_fn(batch):
    """
       data: is a list of tuples with (wav, label)
             where 'wav' is a 1d tensor of arbitrary shape
    """
    waveforms, labels = zip(*batch)
    input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([wav.shape[0] for wav in waveforms]),
            dim=0, descending=True)
    max_input_len = input_lengths[0]

    # process first elem of batch to compute the shape of features and reserve batch size
    win_size_smpl = config['frame_length']*config['sample_rate']//1000
    hope_size_smpl = config['window_shift']*config['sample_rate']//1000
    padded_wav = torch.nn.functional.pad(waveforms[0], pad=(0, max_input_len-waveforms[0].shape[0]))
    first_seq = padded_wav.unfold(0, size=win_size_smpl, step=hope_size_smpl)

    batch_seq = torch.LongTensor(len(batch), *first_seq.shape)
    for i in range(len(waveforms)):
        wav = batch[i][0]
        padded_wav = torch.nn.functional.pad(wav, pad=(0, max_input_len - wav.shape[0]))
        input_seq = padded_wav.unfold(0, size=win_size_smpl, step=hope_size_smpl)
        batch_seq[i] = input_seq
    return batch_seq, labels