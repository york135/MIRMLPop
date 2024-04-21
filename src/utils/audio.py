import librosa



def load_audio_file(file):
    batch = {}
    speech, _ = librosa.load(file, sr=16000)
    # batch["speech"] = resampler.forward(speech.squeeze(0)).numpy()
    # batch["sampling_rate"] = resampler.new_freq
    batch["speech"] = speech
    batch["sampling_rate"] = 16000
    return batch

def load_MIR1k_audio_file(file, mixture: bool=True):
    batch = {}
    speech, _ = librosa.load(file, sr=16000, mono=False)
    # batch["speech"] = resampler.forward(speech.squeeze(0)).numpy()
    # batch["sampling_rate"] = resampler.new_freq
    # (0 + 1) / 2 => mixture
    # 1 => voice
    if mixture:
        batch["speech"] = (speech[0] + speech[1]) / 2
    else:
        batch["speech"] = speech[1]
    batch["sampling_rate"] = 16000
    return batch