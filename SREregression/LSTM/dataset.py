"""Dataset class for LSTM regression model"""
import os
from torch.utils.data import Dataset
import torch
import torchaudio
import glob
from config import config
from augmentation import NoiseAugmentation


class AudioDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.audio_file_paths = glob.glob(os.path.join(data_dir, '**/*.wav'), recursive=True)
        self.transform = transform

    def __len__(self):
        return len(self.audio_file_paths)
        
    def __getitem__(self, idx):
        file_path = self.audio_file_paths[idx]
        audio, sample_rate = torchaudio.load(file_path)
        audio = torch.squeeze(audio)

        #  pad silence from the beggining
        sil_length = 0.02
        audio = torch.nn.functional.pad(audio, pad=(int(0.01*16000), 0))

        if self.transform:
            audio = self.transform(audio)

        # normalization
        mean = torch.mean(audio)
        std = torch.std(audio)

        audio = audio - mean
        audio = audio / std
        
        file_name = file_path.split(os.path.sep)[-1]
        label = file_name.split('.')[0].split('_')[0]
        label = torch.tensor(int(label), dtype=torch.float32)
        return audio, label
    
    
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

    batch_seq = torch.FloatTensor(len(batch), *first_seq.shape)
    for i in range(len(waveforms)):
        wav = batch[i][0]
        padded_wav = torch.nn.functional.pad(wav, pad=(0, max_input_len - wav.shape[0]))
        input_seq = padded_wav.unfold(0, size=win_size_smpl, step=hope_size_smpl)
        batch_seq[i] = input_seq
    return batch_seq, torch.stack(list(labels), dim=0).view(-1, 1)


# # noise augmntation testing

# if __name__ == '__main__':
#     train = AudioDataset(data_dir='/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/LibriSpeechChuncked_sil_removed/train-clean-100',
#                  transform=NoiseAugmentation(noise_dir='/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/ESC-50_16khz/audio'))

#     for x in train:
#         print(x[0])
#         print(x[1])
#         break
    # from scipy.io.wavfile import write

    # os.makedirs('./test', exist_ok=True)
    # for i, x in enumerate(train):
    #     # Assuming x is a tuple where the first element is the audio data
    #     audio_data = x[0].numpy()  # Convert tensor to numpy array
    #     sample_rate = 16000  # Replace with your sample rate
    #     write(f'./test/audio_{i}.wav', sample_rate, audio_data)