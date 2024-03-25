import glob
import os
import torch
import torchaudio
import torchaudio.functional as F


class NoiseAugmentation(torch.nn.Module):
    def __init__(self, noise_dir, noise_prob=0.5):
        super().__init__()
        self.noise_file_paths = glob.glob(os.path.join(noise_dir, '**/*.wav'), recursive=True)
        self.noise_prob = noise_prob

    def __call__(self, audio):
        if torch.rand(1) < self.noise_prob:
            noise_path = self.noise_file_paths[int(torch.randint(0, len(self.noise_file_paths), (1,)))]
            noise, _ = torchaudio.load(noise_path)
            noise = torch.squeeze(noise)
            if noise.shape[0] > audio.shape[0]:
                noise = noise[:audio.shape[0]]
            else:
                noise = F.pad(noise, pad=(0, audio.shape[0]-noise.shape[0]))
            random_snr = torch.randint(0, 21, (1,)).item()
            audio = self.add_noise(audio, noise, random_snr)
        return audio

    def add_noise(self, audio, noise, snr):
        epsilon = torch.tensor(1e-8)
        audio_energy = torch.sum(audio ** 2).item()
        noise_energy = torch.sum(noise ** 2).item()
        
        # Calculate scaling factor for noise to achieve desired SNR
        scaling_factor = torch.sqrt(audio_energy / (10**(snr / 10) * noise_energy + epsilon))
        
        # Add scaled noise to original audio
        noisy_audio = audio + scaling_factor * noise
        return noisy_audio