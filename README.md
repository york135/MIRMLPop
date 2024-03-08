# MIRMLPop

**(2024.03.08)** Since ICASSP 2024 is around the corner, but I haven't had enough time to organize the code for training & evaluation, I decide to release an early version, which only contains the dataset itself (and the code to obtain the audios). Stay tuned for the source code.



This repo is the official repo of the following paper:

Jun-You Wang, Chung-Che Wang, Chon-In Leong, Jyh-Shing Roger Jang, "**MIR-MLPOP: A MULTILINGUAL POP MUSIC DATASET WITH TIME-ALIGNED LYRICS AND AUDIO**," accepted at ICASSP 2024.

It contains 1) annotation of the MIR-MLPop dataset, 2) the source code to obtain the audio of the dataset, 3) source code we used to fine-tune Whisper on MIR-MLPop (both lyrics alignment & lyrics transcription), and 4) source code for evaluation.

Model checkpoints (under *MIR-MLPop + CV* setting) can be found here.

## The MIR-MLPop Dataset

Annotations available at ``dataset/cmn_dataset_240223.json``, `dataset/nan_dataset_240223.json`, and `dataset/yue_dataset_240223.json`.

cmn: Mandarin (Mandarin Chinese / 普通話)

nan: Taiwanese Hokkien (Taiwanese Minnan / 台灣閩南語)

yue: Cantonese (粵語)

**Dataset partition.** song #1~#20 as the training set, #21~#30 as the test set.

**Version.** Currently, the latest version is ``240223``, which is the first version (also, the version for ICASSP 2024) of this dataset.

If there is any annotation error, please contact me (``junyou.wang@mirlab.org``). I may release a newer version in the future. If so, a new version id will also be assigned.

- **Regarding song #25 in ``nan`` subset.** The filler annotation ``[321.112729, 362.658552, "speech"]`` covers the fade out part of the song (after the outro). We (the annotators in the second step) were a little bit unsure whether this fade out part should be considered as lyrics (the volume decreases clearly, the last few characters are almost inaudible; also, the official lyrics do not include this part), so we simply added it into the ``speech`` part. If you think that this part should be a part of the lyrics, then the filler annotation should be modified to ``[321.112729, 323.709000, "speech"]`` (this part is indeed a speech segment), and the lyrics annotation should be added with `送予你的歌 掛念毋敢講出聲 聲聲句句做你拍拚做你去追夢 望你袂虛華 發心袂重耽 擔蔥作穡 實在拚心內較輕鬆 送予你的歌 歌詞藏喙內底咬` (for clarity, here I add blanks to separate phrases).

### Obtain audio files

```
python download_audio.py [json_path] [output_dir]
```

This will download audios included in ``json_path`` to the directory ``output_dir``.

The file name follows the ``song_id`` attribute.

If there is any difficult obtaining audios, please contact me (`junyou.wang@mirlab.org`). In the e-mail, you should at least provide your name and affiliation (if any), and promise that you will only use the audios for academic purpose. Then, I will decide whether to grant you the access or not.

## Preprocessing

### Mute fillers & chunking (to 30-second segments)

### Regarding the Common Voice corpus

Download it.

## Model training: Whisper medium

**Credit:** [openai/whisper](https://github.com/openai/whisper).

r

## Evaluation

## Acknowledgement & contribution of each author

## Cite this work (bibtex)
