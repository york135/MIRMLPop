import argparse
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import json
import whisper
import torch
import pandas as pd
from typing import List, Optional, Any
from pathlib import Path
from tqdm import tqdm

from transformers import AutoTokenizer
import whisper

import random
import numpy as np

from module.align_model import AlignModel
from data_processor.record import Record, read_data

from utils.audio import load_audio_file, load_MIR1k_audio_file

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', "--test-data",
        type=str,
        required=True,
        help="Data file for decode / transcription"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Whisper model name or path"
    )

    parser.add_argument(
        '--model-name',
        default='best'
    )

    parser.add_argument(
        "--suppress-token",
        type=str,
        default=None,
    )

    parser.add_argument(
        '--use-sup-token',
        action='store_true'
    )

    parser.add_argument(
        '--use-pretrained',
        action='store_true'
    )
    parser.add_argument(
        "--language",
        type=str,
        default='zh',
        help="Transcribe language"
    )
    parser.add_argument(
        '--beam_size',
        type=int,
        default=5
    )
    parser.add_argument(
        '--get-timestamps',
        action='store_true'
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=""
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/result.json"
    )

    args = parser.parse_args()

    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def transcribe(
    model: whisper.Whisper,
    records: List[Record],
    get_timestamps: bool=False,
    language: str='zh',
    beam_size: int=50,
    suppress_tokens=[],
    use_sup_token=True,
) -> List[dict]:
    transcribe_results = []
    print ('Use suppress_token?', use_sup_token)
    with torch.no_grad():
        for record in tqdm(records):
            audio_path = record.audio_path
            song_id = Path(audio_path).stem

            audio = load_audio_file(audio_path)['speech']

            # decode_options = {}
            # decode_options['suppress_tokens'] = suppress_tokens
            if use_sup_token:
                result = model.transcribe(audio=audio,
                                          task='transcribe',
                                          language=language,
                                          beam_size=beam_size,
                                          suppress_tokens=suppress_tokens,
                                          )
            else:
                result = model.transcribe(audio=audio,
                                      task='transcribe',
                                      language=language,
                                      beam_size=beam_size,
                                      )

            # print (result['segments'])
            
            if get_timestamps:
                lyric_onset_offset = record.lyric_onset_offset

                inference_onset_offset = []
                for segment in result['segments']:
                    inference_onset_offset.append([segment['start'], segment['end']])
            else:
                lyric_onset_offset = None
                inference_onset_offset = []

            print (record.text)
            print (result['text'])

            transcribe_results.append({
                                        'audio_path': audio_path,
                                        'song_id': song_id,
                                       'lyric': record.text,
                                       'gt_pronounce': record.pronounce,
                                       'inference': result['text'],
                                       'onset_offset': record.lyric_onset_offset,
                                       'inference_onset_offset': inference_onset_offset})

    return transcribe_results

WHISPER_DIM = {'tiny': 384,
               'base': 512,
               'small': 768,
               'medium': 1024,
               'large': 1280}

def load_align_model(
    model_dir: str,
    args,
    device: str='cuda'
) -> AlignModel:
    assert os.path.exists(model_dir)
    with open(os.path.join(model_dir, 'args.json'), 'r') as f:
        train_args = json.load(f)
    tokenizer_name = train_args['tokenizer']
    whisper_model_name = train_args['whisper_model']
    model_path = os.path.join(model_dir, args.model_name + '.pt')
    print (model_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    whisper_model = whisper.load_model(whisper_model_name, device=device)

    if os.path.exists(os.path.join(model_dir, 'model_args.json')):
        with open(os.path.join(model_dir, 'model_args.json'), 'r') as f:
            model_args = json.load(f)
    else:
        model_args = {'embed_dim': WHISPER_DIM[whisper_model_name],
                      'hidden_dim': 384,
                      'bidirectional': True,
                      'output_dim': len(tokenizer) + args.predict_sil,}

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

def main():
    overwrite_output = False
    args = parse_args()
    if os.path.exists(args.output) and overwrite_output == False:
        print("File Exists, Pass")
        exit()

    set_seed(114514)

    device = args.device
    if device == 'cuda' and torch.cuda.is_available() != True:
        device = 'cpu'

    if os.path.exists(args.model_dir) and args.use_pretrained != True:
        transcribe_model = load_align_model(model_dir=args.model_dir,
                                            args=args,
                                            device=device).whisper_model
    else:
        print('Use pretrained model')
        transcribe_model = whisper.load_model(name='medium',
                                              device=device)
    
    transcribe_model.to(device)
    transcribe_model.eval()

    assert os.path.exists(args.test_data)
    test_records = read_data(args.test_data)

    if args.suppress_token is not None:
        with open(args.suppress_token) as json_data:
            suppress_tokens = json.load(json_data)
    else:
        suppress_tokens = None
    
    transcribe_results = transcribe(model=transcribe_model,
                                    records=test_records,
                                    get_timestamps=args.get_timestamps,
                                    language=args.language,
                                    beam_size=args.beam_size,
                                    suppress_tokens=suppress_tokens,
                                    use_sup_token=args.use_sup_token)
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(transcribe_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()