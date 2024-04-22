# MIRMLPop

**(2024.04.23)**: Slightly modified readme & requirements.txt & add the remaining Common Voice subsets.

**(2024.04.22)**: Upload source code (still lack some Common Voice subsets).

**(2024.03.08)**: Upload dataset annotation.

This repo is the official repo of the following paper:


J.-Y. Wang, C.-C. Wang, C.-I. Leong and J.-S. R. Jang, "**MIR-MLPop: A Multilingual Pop Music Dataset with Time-Aligned Lyrics and Audio**," *ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, Seoul, Korea, Republic of, 2024, pp. 1366-1370.

It contains 1) annotation of the MIR-MLPop dataset, 2) the source code to obtain the audio of the dataset, 3) source code we used to fine-tune and evaluate Whisper on MIR-MLPop.

Hope this dataset would facilitate future lyrics transcription & alignment works! Also, big thanks to all of you who came to our poster and discussed with me in the ICASSP 2024 conference!

## The MIR-MLPop Dataset

**(Important)** All the contents in this dataset **cannot** be used for commerical usage.

Annotations available at ``dataset/cmn_dataset_240223.json``, `dataset/nan_dataset_240223.json`, and `dataset/yue_dataset_240223.json`.

cmn: Mandarin (Mandarin Chinese / 普通話)

nan: Taiwanese Hokkien (Taiwanese Minnan / 台灣閩南語)

yue: Cantonese (粵語)

**Dataset partition.** song #1~#20 as the training set, #21~#30 as the test set.

**Version.** Currently, the latest version is ``240223``, which is the first version (also, the version for ICASSP 2024) of this dataset.

**License.** This repo is not allowed for commercial usage. Academic usage is OK.

**Regarding the pronunciation.** For the pronunciation, suppose the correct pronunciation (based on dictionary) is `a`, while the singer actually sings `b`, then the annotation will be `a(b)`.

In our experiments, if such a case occurs, we choose the latter pronunciation (`b`).

There is another case where a singer may fuse two syllables together (I think it is also referred to as "fusion", see https://en.wikipedia.org/wiki/Fusion_(phonetics) and https://zh.wikipedia.org/wiki/閩南語合音, but I'm not very sure). In this case, we annotate the pronunciations as `a((b))` and `c((b))` (suppose the syllable `a` and `c` are fused, becoming `b`).

In our experiments, if such a case occurs, we choose the former pronunciation (`a` and `c`), as such a "fusion" case may introduce new "syllables" which are not really single "syllables".

**Reporting errors.** If there is any annotation error, please contact me (``junyou.wang@mirlab.org``). I may release a newer version in the future. If so, a new version id will also be assigned.

- **Regarding song #25 in ``nan`` subset.** The filler annotation ``[321.112729, 362.658552, "speech"]`` covers the fade out part of the song (after the outro). We were a little bit unsure whether this fade out part should be considered as lyrics (the volume decreases clearly, the last few characters are almost inaudible; also, the official lyrics do not include this part), so we simply added it into the ``speech`` part. If you think that this part should be a part of the lyrics, then the filler annotation should be modified to ``[321.112729, 323.709000, "speech"]`` (this part is indeed a speech segment), and the lyrics annotation should be added with `送予你的歌 掛念毋敢講出聲 聲聲句句做你拍拚做你去追夢 望你袂虛華 發心袂重耽 擔蔥作穡 實在拚心內較輕鬆 送予你的歌 歌詞藏喙內底咬` (for clarity, here I add blanks to separate phrases).

## Install dependencies
```
pip install -r requirements.txt 
```

## Obtain audio files

```
python download_audio.py [json_path] [output_dir]
```

This will download audios included in ``json_path`` to the directory ``output_dir``.

The file name follows the ``song_id`` attribute.

If there is any difficult obtaining audios, please contact me (`junyou.wang@mirlab.org`). In the e-mail, you should at least provide your name and affiliation (if any), and promise that you will only use the audios for academic purpose. Then, I will decide whether to grant you the access or not.

## Preprocessing

### Build dictionary

```
cd preprocessing/
python generate_dict.py [json_path] [output_dict_path]
```

This generates a dictionary at `output_dict_path`. It will be used to assign class id for each syllable (for lyrics alignment).

### Run HT Demucs

**Credit:** [facebookresearch/demucs](https://github.com/facebookresearch/demucs).

```
cd preprocessing/
python demucs_dataset.py [audio_dir] [separated_dir]
```

It separates all subsets in [audio_dir]. By default, it runs HT Demucs on 'cuda'.

If music source separation is not needed, this step can be ignored.

### Mute fillers/speech

```
cd preprocessing/
python remove_filler.py [json_path] [audio_dir] [nofiller_output_dir]
```

Each time it processes one subset. It requires a json file (specified by `[json_path]`) to provide filler information.

### Chunking (convert to 30-second segments)

```
cd preprocessing/
python preprocess_dataset.py -j [json_path] -u [unprocessed_audio_dir] -p [processed_audio_dir] \
    -train [train_json_path] -valid [valid_json_path] -test [test_json_path]
```

Each time it processes one subset. It requires a json file (specified by `[json_path]`) to provide filler information.

Note that we only divide the training data (further separated into a 90/10 train/valid split) into 30-second chunks. For the test data, we do not perform this step. Therefore, `processed_audio_dir` will only contain the separated chunks of 20-song training data.

This script also generates three json files (`train_json_path`, `valid_json_path`, and `test_json_path`), which are the training, validation, and test set, respectively.

### Regarding the Common Voice corpus

To reproduce our experiments with the Common Voice corpus, first, please download the datasets from:

- `cmn`: https://commonvoice.mozilla.org/zh-TW/datasets
- `nan`: https://commonvoice.mozilla.org/nan-tw/datasets
- `yue`: https://commonvoice.mozilla.org/zh-HK/datasets

for all datasets, download the "Common Voice Corpus 14.0" version (2023/06/28).

We provide json files we used for training/validation/testing in `for_common_voice`:

- `cmn`: `cv_zh_tw_train_aug.json`, `cv_zh_tw_valid_aug.json`, `cv_zh_tw_test.json`
- `nan`: `cv_nan_tw_train_aug.json`, `cv_nan_tw_valid_aug.json`, `cv_nan_tw_test.json`
- `yue`: `cv_yue_train_aug.json`, `cv_yue_valid_aug.json`, `cv_yue_test.json`

For `nan`, the `other.tsv` is also used (in order to reach 9 hours of data), while for other languages, we only use the `train.tsv` and `valid.tsv`.

To replicate the experiments, one has to randomly mix the speech data with the Musdb18 dataset's test set:

```
cd for_common_voice/
python mix_with_musdb.py [audio_dir] [augment_dir] [musdb_dir] -5
```

`audio_dir`: The directory to the audio files to be augmented (e.g., `cv-corpus-14.0-2023-06-23/nan-tw/clips`).

``augment_dir``: The directory that the augmented audios will be written to.

``musdb_dir``: The directory to MUSDB-18's test set. It will find those ``accompaniment.wav`` files for each song (in each folder) and randomly choose one for augmentation for each audio. If you don't have such a file, you can sum up the `drum`, `bass`, and `other` tracks to obtain the accompaniments, and store it at `accompaniment.wav`.

(The final argument specifies the desired SNR, which is fixed at -5 in our experiments, i.e., the volume of the accompaniments is about 3 times louder than the speech sound.)

Then, follow the `Run HT Demucs` instruction to apply HT Demucs to the augmented speech data.

Finally, run this script to change the audio path (`song_path`) of each json file a to the desired directory:

```
cd for_common_voice/
python replace_path.py [json_path] [output_json_path] [target_audio_dir]
```

`target_audio_dir`: The directory to the audio files (e.g., `cv-corpus-14.0-2023-06-23/nan-tw/clips_aug_demucs`).

## Model training: Whisper medium

**Credit:** [openai/whisper](https://github.com/openai/whisper).

We choose Whisper medium as a backbone model, and train both lyrics alignment and lyrics transcription in a multitask learning manner.

Basically, the source code here is similar to our previous work on Mandarin lyrics alignment and transcription (see https://github.com/navi0105/LyricAlignment for more information). But we do make some modifications.

```
cd src/
python train_icassp_ver.py --train-data [train json path1] [train json path2]... \
                            --dev_data [valid json path1] [valid json path2] ... \
                            --lr [learning rate] \
                            --save-dir [save dir] \
                            --dict-file [dict file] \
                            --device [device] \
                            [--train-alignment]
```

where `--train-data` specifies the training json files. This could include multiple json files. During training, different datasets will be sampled with equal probability. `--save-dir` specifies the directory to save models; --dict-file specifies the path to the dictionary file (obtained from running `generate_dict.py`). `--train-alignment` specifies whether to train the model with lyrics alignment task or not (store_true). For example, when only Common Voice is used, `--train-alignment` should not be used because there is no data to train lyrics alignment.


`lr` (learning rate) is set to `1e-3` in our experiments. This means that the RNN on top of Whisper is trained with `1e-3` learning rate, and the remainder of the model is trained with `1e-5` learning rate (100 times smaller).

## Evaluation

### Lyrics alignment

```
cd src/
python inference_align_pronun.py --test-data [test json path] --model-dir [model dir] --device [device] \
          --dict-file [dict file] --model-name [model name]
```

In our experiments, we choose the last checkpoint for evaluation, i.e., the `model name` is set to `step4000`.

### Lyrics transcription
During lyrics transcription inference, a small trick is employed: we build a `suppress_token` list. With this list, we force the model to only generate tokens that exist in the training set. We empirically found that this works well when the scale of available training data is small (in this case, the model may not be completely adapted to the new language and may generate tokens that do not belong to the new language).

Therefore, in out experiments, we first build such a list by:

```
cd src/
python get_suppress_token.py --train-data [train json path1] [train json path2]... \  
          --output-path [output path]
```

This will encode all the labels in the training data (you should also include the validation data here) to Whisper tokens. Then, it will generate a list of tokens that do not appear in the training data to `output path`.

Then, run:

```
cd src/
python inference_transcript.py --test-data [test json path] --model-dir [model dir] --device [device] \
          --output [output path] --model-name [model name] --suppress-token [suppress token file] [--use-sup-token]
```

It will generate the transcription results to `[output path]`. In our experiments, we set the `--use-sup-token` flag, which will pass the list stored in `[suppress token file]` as the `suppress_tokens` parameter to Whisper's `transcribe` function.

Finally, as a reference, we always choose the last model checkpoint (`last_model.pt`) for a better performance (regardless of the validation results), which we found to produce slightly better results. Therefore, in practice, it is possible to also use the validation sets for training, and does not use validation set.


As for evaluation:
```
cd src/
python evaluate_transcript.py --result-file [test json path]
```
where `[test json path]` is the result file produced by `inference_transcript.py`.



## Acknowledgement & contribution of each author

We thank the following people for annotating and reviewing the MIR-MLPop dataset:

- **cmn** (Mandarin Chinese) subset: Yu-Teng Hsu, Sean Chang, Michael Zhu, Bo-Han Feng, Hsueh-Han Lee, Chen Yi Liang, and Yu-Chun Lin.

- **yue** (Cantonese) subset: Kin Fong Chao, Choi Tim Ho, Charlotte Ong, Wing
  Tung Yeung, and Chak Chon Pun.

- **nan** (Taiwanese Hokkien) subset: Yu-Chun Lin, Liau Sui-Lin, Zhe-Hao Lin, Wei-Chieh Chou, and Meng-Hua Yu.



As for the contribution of each author:

- Jun-You Wang (github account: @york135) was responsible for curating the nan subset and a part of the cmn subset. He also organized this repo and conducted the experiments for the nan and cmn subset.

- Chung-Che Wang was responsible for curating a large part of the cmn subset.

- Chon In Leong (github account: @navi0105) was responsible for curating the yue subset and conducting the experiments for the yue subset.



## Cite this work

#### Bibtex

@INPROCEEDINGS{Wang2024mirmlpop,
  author={Wang, Jun-You and Wang, Chung-Che and Leong, Chon-In and Jang, Jyh-Shing Roger},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={MIR-MLPop: A Multilingual Pop Music Dataset with Time-Aligned Lyrics and Audio}, 
  year={2024},
  volume={},
  number={},
  pages={1366-1370},
  doi={10.1109/ICASSP48485.2024.10447561}}

#### Plain text

J.-Y. Wang, C.-C. Wang, C.-I. Leong and J.-S. R. Jang, "MIR-MLPop: A Multilingual Pop Music Dataset with Time-Aligned Lyrics and Audio," *ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, Seoul, Korea, Republic of, 2024, pp. 1366-1370.
