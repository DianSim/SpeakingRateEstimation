import os
import soundfile as sf


def Speed(dir, save_dir_name, speed_factor):
    replace_name = dir.split(os.sep)[-2]
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file[-4:] == '.wav':
                signal, sr = sf.read(os.path.join(root, file))

                save_dir = root.replace(replace_name, save_dir_name)
                os.makedirs(save_dir, exist_ok=True)
                sf.write(os.path.join(save_dir, file), signal, int(sr*speed_factor))

                print('file:', os.path.join(save_dir, file))
                print('speed factor: ', speed_factor)


dir = '/data/saten/diana/SpeakingRateEstimation/data/LibriSpeechChuncked_train_3sec/train-clean-100'
Speed(dir, 'LibriSpeechChunked_train_fast_1_5x_24khz', speed_factor=1.5)