import os
import soundfile as sf
import pandas as pd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import math
import numpy as np


# /train-clean-100/481/123720/3_11.wav - 2.8000625

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


def removing_silence(dir):
    """"
    Splits the audios of the given dir into chunks, computes the labels
    and saves in a folder 'LibriSpeechChuncked' with the same structure as dir
    setting lables as audio titles

    chunck_length: chunk length with seconds
    """
    for root, dirs, files in os.walk(dir):
        word_trans_file = None
        for file in files:
            if file[-14:] == '.alignment.txt':
                word_trans_file = os.path.join(root, file)
        if word_trans_file is None:
            continue

        folder_file_title_count = {}
        df_word_trans = pd.read_csv(word_trans_file, delimiter='\t', header=None)
        for k, file in enumerate(files):
            if file[-5:] == '.flac':
                signal, sr = sf.read(os.path.join(root, file))
                for row in df_word_trans[0]:
                    if row.split(' ')[0] == file.split('.')[0]:
                        words_str = row.split(' ')[1]
                        words_str = words_str.replace('"', '')
                        tm_stemps_str = row.split(' ')[2]
                        tm_stemps_str = tm_stemps_str.replace('"', '')
                        words = words_str.split(',')
                        tm_stemps = tm_stemps_str.split(',')
                        tm_stemps = [float(t) for t in tm_stemps]

                        print(words)
                        print(tm_stemps)
                        print()
                        print('-'*60+'chunking'+'-'*60)
                        print()

                        silence_removed_tms = list(tm_stemps)
                        for i, word in enumerate(words):
                            if word == '':
                                sil_len = silence_removed_tms[i] - (silence_removed_tms[i-1] if i else 0)
                                # shift words after silence by silence length
                                for j in range(i+1, len(silence_removed_tms)):
                                    silence_removed_tms[j] -= sil_len

                        i_silence = [i for i, word in enumerate(words) if word=='']

                        if len(i_silence) == 0:
                            x = signal
                        else:
                            sil_begin = tm_stemps[i_silence[0]-1] if i_silence[0] else 0
                            splits = [signal[0: round(sil_begin*sr) + 1]]
                            for i in range(len(i_silence)):
                                if i+1 < len(i_silence):
                                    # from end of 1st sil to beginning of 2nd
                                    i_sil_left = i_silence[i]
                                    i_sil_right = i_silence[i+1]
                                    splt = signal[math.floor(tm_stemps[i_sil_left]*sr): round(tm_stemps[i_sil_right -1]*sr) + 1]
                                    splits.append(splt)

                            last_splt = signal[math.floor(tm_stemps[i_silence[-1]]*sr): round(tm_stemps[1]*sr) + 1]
                            splits.append(last_splt)
                            x = np.concatenate(splits)
                        
                        save_dir = root.replace('test-clean', 'test-clean-sil-removed')
                        os.makedirs(save_dir, exist_ok=True)

                        transcript = ' '.join(words)
                        speak_rate = speaking_rate(transcript)

                        print(os.path.join(save_dir, f'{k}_{speak_rate}.wav'))
                        print(x)
                        sf.write(os.path.join(save_dir, f'{k}_{speak_rate}.wav'), x, sr)

                        print('transcript: ', transcript)
                        print('syl_count: ', speak_rate)
           
                        # i_begin = 0
                        # while i_begin < len(words):
                        #     i_end = i_begin

                        #     # chunck begin is the end of the first non silence words' end
                        #     if i_begin == 0:
                        #         chunck_begin = 0
                        #     else:
                        #         i_non_sil_word = i_begin - 1
                        #         # if this is the second chunck there should be non silence word
                        #         while words[i_non_sil_word] == '':
                        #             i_non_sil_word -= 1
                        #         chunck_begin = silence_removed_tms[i_non_sil_word] if i_begin else 0

                        #     chunck_end = chunck_begin + chunck_length

                        #     while i_end < len(words):
                        #         if words[i_end] == '' or (words[i_end] != '' and silence_removed_tms[i_end] <= chunck_end):
                        #             i_end += 1
                        #         else:
                        #             break

                        #     if i_end != i_begin:
                        #         i_end -= 1 # for i_end the condition deosn't hold

                        #     chunk_words = words[i_begin:i_end+1]

                        #     transcript = ' '.join(chunk_words)

                        #     speak_rate = speaking_rate(transcript)
                        #     print('sr:', speak_rate)

                        #     i_silence = [i for i in range(i_begin, i_end+1) if words[i]=='']
                        #     chunck_begin = tm_stemps[i_begin-1] if i_begin else 0

                        #     if len(i_silence) == 0:
                        #         x = signal[math.floor(chunck_begin*sr): round(tm_stemps[i_end]*sr) + 1]
                        #     else:
                        #         sil_begin = tm_stemps[i_silence[0]-1] if i_silence[0] else 0
                        #         splits = [signal[math.floor(chunck_begin*sr): round(sil_begin*sr) + 1]]
                        #         for i in range(len(i_silence)):
                        #             if i+1 < len(i_silence):
                        #                 # from end of 1st sil to beginning of 2nd
                        #                 i_sil_left = i_silence[i]
                        #                 i_sil_right = i_silence[i+1]
                        #                 splt = signal[math.floor(tm_stemps[i_sil_left]*sr): round(tm_stemps[i_sil_right -1]*sr) + 1]
                        #                 splits.append(splt)

                        #         last_splt = signal[math.floor(tm_stemps[i_silence[-1]]*sr): round(tm_stemps[i_end]*sr) + 1]
                        #         splits.append(last_splt)
                        #         x = np.concatenate(splits)

                        #     # pad silence in the beginning
                        #     silence_len = int(0.02*16000) # should be determined somehow
                        #     x = np.pad(x, (silence_len, 0), 'constant', constant_values=0)
                        #     exact_chunck_len = x.shape[0]/sr


                        #     # save_dir = root.replace('LibriSpeech', 'LibriSpeechChuncked_sil_removed')

                        #     save_dir_4 = root.replace('LibriSpeech', 'LibriSpeechChuncked_train_4sec')
                        #     save_dir_3 = root.replace('LibriSpeech', 'LibriSpeechChuncked_train_3sec')

                        #     if not os.path.exists(save_dir_4):
                        #         os.makedirs(save_dir_4)

                        #     if not os.path.exists(save_dir_3):
                        #         os.makedirs(save_dir_3)

                        #     # if not os.path.exists(save_dir):
                        #     #     os.makedirs(save_dir)

                        #     filename = str(speak_rate)
                        #     if filename not in folder_file_title_count:
                        #         folder_file_title_count[filename] = 0
                        #     else:
                        #         folder_file_title_count[filename] += 1
                        #         filename += f'_{folder_file_title_count[filename]}'
                        #     filename += '.wav'

                        #     print('transcripts: ', transcript)
                        #     print(f'exact chunck length: {exact_chunck_len}')

                        #     # print(os.path.join(save_dir, filename))
                        #     # sf.write(os.path.join(save_dir, filename), x, sr)

                        #     if exact_chunck_len >= 3.8:
                        #         print(os.path.join(save_dir_4, filename))
                        #         sf.write(os.path.join(save_dir_4, filename), x, sr)
                        #     elif exact_chunck_len > 2.8 and exact_chunck_len < 3.3:
                        #         print(os.path.join(save_dir_3, filename))
                        #         sf.write(os.path.join(save_dir_3, filename), x, sr)

                        #     print()
                        #     # if exact_chunck_len > 2.5:
                        #     #     print(os.path.join(save_dir, filename))

                        #     i_begin = i_end + 1


if __name__ == '__main__':
    dir = '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/LibriSpeech/test-clean'
    removing_silence(dir)