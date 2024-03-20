import os
import soundfile as sf
import numpy as np
import pandas as pd
import shutil


def Speed(dir, save_dir_name):
    for root, dirs, files in os.walk(dir):
        trans_file = None
        for file in files:
            if file[-10:] == '.trans.txt':
                trans_file = os.path.join(root, file)

        if trans_file is None: # means there're no audios in the folder
            continue

        df_trans_file = pd.read_csv(trans_file, delimiter='\t', header=None)
        # print(df_trans_file[0].head())

        for file in files:
            if file[-5:] == '.flac': 
                signal, sr = sf.read(os.path.join(root, file))
                print(signal.shape)
                speed_factor = np.random.uniform(1.5, 2)
                save_dir = root.replace('LibriSpeech', save_dir_name)
                os.makedirs(save_dir, exist_ok=True)
                sf.write(os.path.join(save_dir, file.split('.')[0] + '.wav'), signal, int(sr*speed_factor))

                print('file:', os.path.join(save_dir, file.split('.')[0] + '.wav'))
                print('speed factor: ', speed_factor)
                for row in df_trans_file[0]:
                    if row.split()[0] == file.split('.')[0]:
                        transcript = ' '.join(row.split()[1:])
                        print(f'transcript: {transcript}')
                        print()
        
        dest = root.replace('LibriSpeech', save_dir_name)
        os.makedirs(dest, exist_ok=True)
        shutil.copy(trans_file, root.replace('LibriSpeech', save_dir_name))
                

dir = '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/LibriSpeech/train-clean-100'
Speed(dir, 'LibriSpeech_Fast')