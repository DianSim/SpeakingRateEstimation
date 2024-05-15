import os
import shutil

def merge(dir):
    """Merges word based allignments in the given directory(dir)
    with Librispeech dataset. Allignments' dir's structure
    is the same as dataset's structure"""
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file[-4:] == '.txt':
                soure = os.path.join(root, file)
                components = soure.split(os.path.sep)
                components.remove('LibriSpeech-Alignments')
                components.remove(file)
                dst_dir = os.path.sep.join(components)
                if os.path.exists(dst_dir):
                    shutil.copy(soure, dst_dir)


allignment_dir = '/home/diana/Desktop/MyWorkspace/Project/SpeakingRateEstimation/data/LibriSpeech-Alignments'
if __name__ == '__main__':
    merge(allignment_dir)
