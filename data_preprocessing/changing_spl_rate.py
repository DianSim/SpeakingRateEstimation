import os
import subprocess
import tqdm
import soundfile as sf



def change_sample_rate(dir, sample_rate, out_dir_name):
    replace_name = dir.split(os.sep)[-2]
    for root, dirs, files in os.walk(dir):
        save_dir = root.replace(replace_name, out_dir_name)
        os.makedirs(save_dir, exist_ok=True)
        for file in tqdm.tqdm(files):
            command = f"ffmpeg -i '{os.path.join(root, file)}' -ar {sample_rate} '{os.path.join(save_dir, file)}'"
            result = subprocess.run(['ffmpeg', '-i', f'{os.path.join(root, file)}', '-ar', f'{sample_rate}', f'{os.path.join(save_dir, file)}'], capture_output=True)


def sample_rate_list(dir):
    """returns the set of sample rates of the audios in the given folder"""
    sample_rates = set()
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file[-4:] == '.wav':
                signal, sr = sf.read(os.path.join(root, file))
                sample_rates.add(sr)
    return sample_rates


# source = '/data/saten/diana/SpeakingRateEstimation/data/ESC-50-master/audio'
# change_sample_rate(source, 16000, 'ESC-50_16khz')


# --------------------------------Testing--------------------------------
print(sample_rate_list('/data/saten/diana/SpeakingRateEstimation/data/LibriSpeechChuncked_sil_removed'))