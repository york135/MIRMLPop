import csv
import os, sys
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import statistics
import matplotlib.pyplot as plt

import librosa
from tqdm import tqdm
import soundfile as sf


if __name__ == "__main__":
    json_path = sys.argv[1]
    audio_dir = sys.argv[2]
    # test_nofiller_audio_dir = '../nan_tw_demucs_test_nofiller'
    nofiller_output_dir = sys.argv[3]

    if not os.path.exists(nofiller_output_dir):
        os.mkdir(nofiller_output_dir)

    with open(json_path) as json_data:
        data = json.load(json_data)

    print (len(data))

    training_set = []
    test_set = []

    for i in tqdm(range(len(data))):
        wav_path = os.path.join(audio_dir, data[i]['song_id'] + '.wav')
        
        song_id = os.path.basename(wav_path)
        wav_data, _ = librosa.load(wav_path, sr=44100)

        filler_label = data[i]['filler']
        for j in range(len(filler_label)):
            start = int(round(filler_label[j][0] * 44100))
            end = int(round(filler_label[j][1] * 44100))
            wav_data[start:end] = 0

        output_path = os.path.join(nofiller_output_dir, data[i]['song_id'] + '.wav')
        sf.write(
            output_path,
            wav_data,
            44100,
            "PCM_16",
        )