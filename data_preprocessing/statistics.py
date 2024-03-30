import os
import soundfile as sf
import matplotlib.pyplot as plt
from collections import Counter


def class_distr(sprs, prefix, folder):
    """plots barplot of speaking rate classes and saves as a picture in the given folder"""
    value_counts = Counter(sprs)
    keys = value_counts.keys()
    values =  value_counts.values()
    plt.figure()
    plt.bar(keys, values)
    plt.xlabel('Sylable count')
    plt.ylabel('Count')
    plt.title(f"{prefix} classes distribution")
    plt.xticks(ticks=range(len(keys)), labels=sorted(keys))
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'{prefix}_class_distr.png'))
    print(value_counts)


def save(arr, xlabel, title, save_path):
    plt.figure()
    plt.hist(arr) #, bins=10) # bins=int(max(arr))-int(min(arr))+1) # ,, color='blue', edgecolor='black')
    # Add labels and title
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(save_path)


def stat(dir, save_dir, split):
    """Computes speaking rate distribution and chunck length distribution
    of the audios in the given folder and saves in the 'save_dir' folder as png

    split: split name (train/val/test)"""

    os.makedirs(save_dir, exist_ok=True)
    lens = []
    spr = []
    len_in_samples = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file[-4:] == '.wav':
                signal, sr = sf.read(os.path.join(root, file))
                len_in_samples.append(len(signal))
                lens.append(len(signal)/sr)
                spr.append(int(file.split('.')[0].split('_')[0]))
    save(lens, 'chunck length', f'{split} set chunk length distribution', os.path.join(save_dir, f'{split}_length'))
    save(spr, 'speaking rate (#vowal)', f'{split} set speaking rate distribution', os.path.join(save_dir, f'{split}_sp_rate'))
    class_distr(spr, split, save_dir)

    with open(os.path.join(save_dir,f'{split}.txt'), 'w') as file:
        file.write(f"max speaking rate({split}): {max(spr)}\n")
        file.write(f"number of audios ({split}): {len(lens)}\n")
        file.write(f"max length ({split}): {max(lens)}\n")
        file.write(f"min length ({split}): {min(lens)}\n")
        file.write(f"max length in samples ({split}): {max(len_in_samples)}\n")



if __name__ == '__main__':
    # save_dir = '/data/saten/diana/SpeakingRateEstimation/data/Statistics/LibriSpeechChuncked_sil_removed'
    # val_dir = '/data/saten/diana/SpeakingRateEstimation/data/LibriSpeechChuncked_sil_removed/dev-clean'
    # test_dir = '/data/saten/diana/SpeakingRateEstimation/data/LibriSpeechChuncked_sil_removed/test-clean'
    # train_dir = '/data/saten/diana/SpeakingRateEstimation/data/LibriSpeechChuncked_sil_removed/train-clean-100'


    # stat(val_dir, save_dir, 'val')
    # stat(test_dir, save_dir, 'test')
    # stat(train_dir, save_dir, 'train')

    dir = '/data/saten/diana/SpeakingRateEstimation/data/LibriSpeechChuncked_sil_removed/train-clean-100-fast-augmented'

    stat(dir, '/data/saten/diana/SpeakingRateEstimation/data/Statistics/LibriSpeechChuncked_sil_removed', 'train-fast-augmented')
    # file = '/data/saten/diana/SpeakingRateEstimation/data/LibriSpeechChunked_train_fast_2x/train-clean-100/19/198/16_3.wav'
    # signal, sr = sf.read(file)
    # print(sr)