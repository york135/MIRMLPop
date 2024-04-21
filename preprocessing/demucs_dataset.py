from demucs import pretrained, apply
import librosa
import numpy as np
import soundfile as sf
import torch
import sys, os
from tqdm import tqdm
import random

device = 'cuda'
model = pretrained.get_model(name="htdemucs").to(device)
model.eval()

def infer_vocal_demucs(mix_np):
    # mix_np is of shape (channels, time)
    mix = torch.tensor([mix_np, mix_np]).float().to(device)
    sources = apply.apply_model(model, mix[None], split=True, overlap=0.5, progress=False)[0]
    return sources[model.sources.index('vocals')].detach().cpu().numpy()


if __name__ == "__main__":
    audio_dir = sys.argv[1]
    separated_dir = sys.argv[2]

    # HT Demucs is actually non-deterministic!
    random.seed(114514)
    np.random.seed(114514)
    torch.manual_seed(114514)
    torch.cuda.manual_seed(114514)

    if not os.path.exists(separated_dir):
        os.mkdir(separated_dir)
    
    for subset_name in tqdm(os.listdir(audio_dir)):
        subset_dir = os.path.join(audio_dir, subset_name)
        output_dir = os.path.join(separated_dir, subset_name)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for audio_name in tqdm(os.listdir(subset_dir)):
            audio_path = os.path.join(subset_dir, audio_name)

            y, _ = librosa.load(audio_path, sr=44100, mono=True)
            output = infer_vocal_demucs(y).T
            output = (output[:,0] + output[:,1]) / 2


            output_path = os.path.join(output_dir, audio_name)
            sf.write(
                output_path,
                output,
                44100,
                "PCM_16",
            )