import os
import soundfile as sf
import pandas as pd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import math
import numpy as np


def speaking_rate(transcript):
    """The function computes syllable based speaking rate
    of an audio with given transcript

    transcript(str): transcript of the audio
    """
    vowels = ['a', 'e', 'i', 'o', 'u', 'y']
    vowel_count = 0
    for s in transcript:
        if s.lower() in vowels:
            vowel_count += 1
    # return vowel_count/length
    return vowel_count


def LibriSpeechLabeling(dir):
    for root, dirs, files in os.walk(dir):
        trans_file = None
        for file in files:
            if file[-10:] == '.trans.txt':
                trans_file = os.path.join(root, file)
        if trans_file is None:
            continue

        df_trans_file = pd.read_csv(trans_file, delimiter='\t', header=None)
        print(df_trans_file.head())  
        for i, file in enumerate(files):
            if file[-5:] == '.flac':
                signal, sr = sf.read(os.path.join(root, file))
                for row in df_trans_file[0]:
                    if row.split(' ')[0] == file.split('.')[0]:
                        transcript = ' '.join(row.split(' ')[1:]).lower()
                        print(transcript)
                        syl_count = speaking_rate(transcript)
                        save_dir = root.replace('test-clean', 'test-clean-labeled')
                        os.makedirs(save_dir, exist_ok=True)
                        sf.write(os.path.join(save_dir, f'{i}_{syl_count}.wav'), signal, sr)
                        print('path: ', os.path.join(save_dir, f'{i}_{syl_count}.wav'))
                        print()

dir = '/home/diana/Desktop/MyWorkspace/Project/SpeakingRateEstimation/data/LibriSpeech/test-clean'
LibriSpeechLabeling(dir)