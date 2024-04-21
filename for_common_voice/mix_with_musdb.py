# from demucs import pretrained, apply
import librosa
import numpy as np
import soundfile as sf
import torch
import sys, os
from tqdm import tqdm

import random
import math

# device = 'cuda:1'
# model = pretrained.get_model(name="htdemucs").to(device)
# model.eval()

# def infer_vocal_demucs(mix_np):
#     # mix_np is of shape (channels, time)
#     mix = torch.tensor([mix_np, mix_np]).float().to(device)
#     sources = apply.apply_model(model, mix[None], split=True, overlap=0.5, progress=False)[0]
#     return sources[model.sources.index('vocals')].detach().cpu().numpy()

if __name__ == "__main__":

    random.seed(114514)

    audio_dir = sys.argv[1]
    augment_dir = sys.argv[2]
    musdb_dir = sys.argv[3]

    snr = float(sys.argv[4])

    print ("SNR:", snr)

    if not os.path.exists(augment_dir):
        os.mkdir(augment_dir)


    accompaniment_pool = []
    for song_name in tqdm(os.listdir(musdb_dir)):
        audio_path = os.path.join(musdb_dir, song_name, 'accompaniment.wav')
        y, _ = librosa.load(audio_path, sr=44100, mono=True)
        y = librosa.util.normalize(y)
        accompaniment_pool.append(y)

    print (accompaniment_pool[0].shape, accompaniment_pool[1].shape)

    # SNR=-5, power=1/sqrt(10) times
    fixed_voc_to_acc_ratio = math.pow(10.0, snr / 10.0)
    print ('Vocal to instrument energy ratio:', fixed_voc_to_acc_ratio)
    
    for audio_name in tqdm(os.listdir(audio_dir)):
        audio_path = os.path.join(audio_dir, audio_name)

        y, _ = librosa.load(audio_path, sr=44100, mono=True)
        y = librosa.util.normalize(y)
        y = y / 2.0

        y_energy_power = np.mean(y ** 2)
        y_dur = len(y)

        clip_id_for_aug = random.randint(0, len(accompaniment_pool) - 1)

        cur_clip_length = accompaniment_pool[clip_id_for_aug].shape[0]

        if cur_clip_length < y_dur:
            print (cur_clip_length, y_dur)

        cur_clip_start_time = random.randint(0, cur_clip_length - y_dur)
        instrument_to_aug = accompaniment_pool[clip_id_for_aug][cur_clip_start_time:cur_clip_start_time + y_dur]
        instrument_to_aug_energy_power = np.mean(instrument_to_aug ** 2)

        # If the selected instruments segment is actually silence, find another one
        while instrument_to_aug_energy_power < 0.0001:
            cur_clip_start_time = random.randint(0, cur_clip_length - y_dur)
            instrument_to_aug = accompaniment_pool[clip_id_for_aug][cur_clip_start_time:cur_clip_start_time + y_dur]
            instrument_to_aug_energy_power = np.mean(instrument_to_aug ** 2)

        instrument_to_aug = librosa.util.normalize(instrument_to_aug)
        instrument_to_aug = instrument_to_aug / 2.0
        instrument_to_aug_energy_power = np.mean(instrument_to_aug ** 2)

        if instrument_to_aug_energy_power * fixed_voc_to_acc_ratio > y_energy_power:
            rescale_acc_factor = y_energy_power / (instrument_to_aug_energy_power * fixed_voc_to_acc_ratio)
            # From power to amplitude
            rescale_acc_factor = rescale_acc_factor ** 0.5
            instrument_to_aug = instrument_to_aug * rescale_acc_factor
        else:
            rescale_voc_factor = (instrument_to_aug_energy_power * fixed_voc_to_acc_ratio) / y_energy_power
            rescale_voc_factor = rescale_voc_factor ** 0.5
            y = y * rescale_voc_factor

        output = y + instrument_to_aug

        output_path = os.path.join(augment_dir, audio_name.split('.')[0] + '.wav')
        # print (output_path)
        # print (output.shape)
        sf.write(
            output_path,
            output,
            44100,
            "PCM_16",
        )