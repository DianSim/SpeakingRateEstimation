import os
import subprocess
import tqdm
import soundfile as sf


# incomplte

# ffmpeg -i input.mp3 -ar 16000 output.wav

def mp3_to_wav(dir, sample_rate, out_dir_name):
    replace_name = dir.split(os.sep)[-1]
    for root, dirs, files in os.walk(dir):
        save_dir = root.replace(replace_name, out_dir_name)
        os.makedirs(save_dir, exist_ok=True)
        for file in tqdm.tqdm(files):
            command = f"ffmpeg -i '{os.path.join(root, file)}' -ar {sample_rate} '{os.path.join(save_dir, file)}'"
            result = subprocess.run(['ffmpeg', '-i', f'{os.path.join(root, file)}', '-ar', f'{sample_rate}', f'{os.path.join(save_dir, file)}'], capture_output=True)
