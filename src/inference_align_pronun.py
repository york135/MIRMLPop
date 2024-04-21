import os
import json
import argparse
import random
import itertools
import numpy as np
from typing import Iterator, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
from pypinyin import lazy_pinyin, Style

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import whisper
from whisper.audio import log_mel_spectrogram
from whisper.tokenizer import get_tokenizer
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from module.align_model import AlignModel
from dataset import get_multitask_dataloader
from utils.alignment import perform_viterbi, get_mae

os.environ["TOKENIZERS_PARALLELISM"]="false"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--test-data',
        type=str,
        required=True
    )

    parser.add_argument(
        '-d', '--dict-file',
        type=str,
        required=True
    )

    parser.add_argument(
        '--model-dir',
        type=str,
        default=None
    )
    parser.add_argument(
        '--model-name',
        default='best_model'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=114514
    )

    args = parser.parse_args()
    return args

WHISPER_DIM = {'tiny': 384,
               'base': 512,
               'small': 768,
               'medium': 1024,
               'large': 1280}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def load_align_model(
    model_dir: str,
    args,
    device: str='cuda'
) -> AlignModel:
    assert os.path.exists(model_dir)
    with open(os.path.join(model_dir, 'args.json'), 'r') as f:
        train_args = json.load(f)
    whisper_model_name = train_args['whisper_model']

    model_path = os.path.join(model_dir, args.model_name + '.pt')
    print (model_path)

    whisper_model = whisper.load_model(whisper_model_name, device=device)

    with open(os.path.join(model_dir, 'model_args.json'), 'r') as f:
        model_args = json.load(f)

    bidirectional = model_args.get('bidirectional', True)

    model = AlignModel(whisper_model=whisper_model,
                       embed_dim=model_args['embed_dim'],
                       hidden_dim=model_args['hidden_dim'],
                       output_dim=model_args['output_dim'],
                       bidirectional=bidirectional,
                       device=device)
    
    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict=state_dict)
    
    return model


@torch.no_grad()
def align_and_evaluate(
    model: AlignModel,
    test_dataloader: DataLoader,
    device: str='cuda'
):
    total_mae = 0
    model.eval()
    model.to(device)
    pbar = tqdm(test_dataloader)
    cnt = 0

    for batch in pbar:
        audios, tokens, _, lyric_word_onset_offset, _, _ = batch

        if lyric_word_onset_offset == (None,) or lyric_word_onset_offset == ([],):
            continue

        align_logits, _ = model.frame_manual_forward(audios)

        align_logits = align_logits.cpu()
        
        align_results = perform_viterbi(align_logits, tokens)
        
        # print(align_logits)
        # print(lyric_word_onset_offset)
        # print(align_results)

        mae = get_mae(lyric_word_onset_offset, align_results)
        print (mae)
        pbar.set_postfix({"current MAE": mae})

        total_mae += mae
        cnt = cnt + 1

    # avg_mae = total_mae / len(test_dataloader)
    avg_mae = total_mae / cnt
    print("Average MAE:", avg_mae)
    return avg_mae

def main():
    args = parse_args()

    # print (args)
    set_seed(args.seed)

    device = args.device
    if 'cuda' in device and torch.cuda.is_available() == False:
        device = 'cpu'

    # Load Tokenizer, Model
    assert os.path.exists(args.model_dir)
    model = load_align_model(args.model_dir, args, device=device)
    whisper_tokenizer = get_tokenizer(multilingual=True, task='transcribe')
    
    assert os.path.exists(args.test_data)

    pronounce_lookup_table = {}
    with open(args.dict_file) as json_data:
        dictionary = json.load(json_data)

    for i in range(len(dictionary['pronounce'])):
        pronounce_lookup_table[dictionary['pronounce'][i]] = i + 1

    test_dataloader = get_multitask_dataloader(args.test_data,
                                               pronounce_lookup_table=pronounce_lookup_table,
                                               whisper_tokenizer=whisper_tokenizer,
                                               batch_size=args.batch_size,
                                               shuffle=False)

    align_and_evaluate(model=model,
                       test_dataloader=test_dataloader,
                       device=device)


if __name__ == "__main__":
    main()
