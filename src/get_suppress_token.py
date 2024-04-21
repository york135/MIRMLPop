import os
import json
import argparse
import random
import numpy as np
from typing import Iterator, Tuple
from tqdm import tqdm
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import whisper
from whisper.tokenizer import get_tokenizer
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from module.align_model import AlignModel
from utils.alignment import get_ce_weight
from dataset import get_multitask_dataloader
from pypinyin import lazy_pinyin, Style


from data_processor.record import read_data

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["TOKENIZERS_PARALLELISM"]="false"

def parse_args():
    parser = argparse.ArgumentParser()
    # Data Argument
    parser.add_argument(
        '--train-data',
        nargs='+',
        type=str,
        required=True
    )

    parser.add_argument(
        '--output-path',
        type=str
    )


    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    whisper_tokenizer = get_tokenizer(multilingual=True, task="transcribe")

    records = []
    for path in args.train_data:
        print (path)
        records.extend(read_data(path))

    valid_output_token = []
    for record in records:
        no_timestamps = True
        transcript_text_tokens = whisper_tokenizer.encode(record.text)
        # print (record.text)
        # print (transcript_text_tokens)
        special_tokens = [
                whisper_tokenizer.sot,
                whisper_tokenizer.special_tokens[f"<|zh|>"],
                whisper_tokenizer.special_tokens["<|transcribe|>"],
            ]

        decoder_input = special_tokens + transcript_text_tokens
        decoder_output = special_tokens[1:] + transcript_text_tokens + [whisper_tokenizer.eot]

        # print (decoder_input)
        # print (decoder_output)

        for i in range(len(decoder_output)):
            if decoder_output[i] not in valid_output_token:
                valid_output_token.append(decoder_output[i])

    if whisper_tokenizer.no_speech not in valid_output_token:
        valid_output_token.append(whisper_tokenizer.no_speech)

    print (valid_output_token, len(valid_output_token))
    print (whisper_tokenizer.encoding.max_token_value)

    suppress_tokens = []
    for i in range(whisper_tokenizer.encoding.max_token_value + 1 - 1501):
        # Allow timestamp token
        if i not in valid_output_token:
            suppress_tokens.append(i)

    with open(args.output_path, 'w') as f:
        json.dump(suppress_tokens, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()