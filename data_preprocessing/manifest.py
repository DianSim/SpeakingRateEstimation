import json

# /Users/dianasimonyan/NeMo

text = "WHO IS SHE I FOLLOWED AT MY HUMBLE DISTANCE THE EXAMPLE OF SIR WALTER SCOTT WHEN INQUISITIVE PEOPLE ASKED HIM IF HE WAS THE AUTHOR OF THE WAVERLEY NOVELS IN PLAIN ENGLISH I DENIED ALL KNOWLEDGE OF THE STRANGER WEARING THE GREEN HAT"

manifest_filepath = f"./manifest.json"
manifest_data = {
    "audio_filepath": '/Users/dianasimonyan/Desktop/Thesis/Implementation/datasets/LibriSpeech_Fast/train-clean-100/5022/29411/5022-29411-0030.wav',
    "text": text
}

with open(manifest_filepath, 'w') as f:
    line = json.dumps(manifest_data)
    f.write(line + "\n")