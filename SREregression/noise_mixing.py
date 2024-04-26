import glob 
import os
import random
import soundfile as sf
import math
import numpy as np
import tqdm


def add_noise(audio, noise, snr):
    epsilon = 1e-8
    audio_energy = np.sum(audio ** 2)
    noise_energy = np.sum(noise ** 2)

    # Calculate scaling factor for noise to achieve desired SNR
    scaling_factor = np.sqrt(audio_energy / (10**(snr / 10) * noise_energy + epsilon))

    # Add scaled noise to original audio
    noisy_audio = audio + scaling_factor * noise
    return noisy_audio


def noise_mixing(data_dir, noise_dir, snr_db):
    """
    This function mixes the given data with noise randomly chosen from noise_dir with given decibel
    """
    noise_file_paths = glob.glob(os.path.join(noise_dir, '**/*.wav'), recursive=True)
    # data_file_paths = glob.glob(os.path.join(data_dir, '**/*.wav'), recursive=True)

    for root, dirs, files in os.walk(data_dir):
        for file in tqdm.tqdm(files):

            i_noise = random.randint(0, len(noise_file_paths)-1)
            rand_noise = noise_file_paths[i_noise]

            file_path = os.path.join(root, file)
            audio, sr = sf.read(file_path)
            noise, sr = sf.read(rand_noise)

            if noise.shape[0] > audio.shape[0]:
                noise = noise[:audio.shape[0]]
            elif noise.shape[0] < audio.shape[0]:
                repeat = math.ceil(audio.shape[0] / noise.shape[0])
                long_noise = np.tile(noise, repeat)
                noise = long_noise[:audio.shape[0]]
            noisy_signal = add_noise(audio, noise, snr_db)

            save_dir = root.replace(root.split(os.sep)[-1], f'noisy_mixed_wavs_{snr_db}db')

            os.makedirs(save_dir, exist_ok=True)
            sf.write(os.path.join(save_dir, file), noisy_signal, sr)



if __name__ == "__main__":
    data_dir = '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/CommonVoice/en/clips_wav_16khz_labeled'
    noise_dir = '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/ESC-50_16khz/audio'
    noise_mixing(data_dir, noise_dir, 10)