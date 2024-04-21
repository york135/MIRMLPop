import argparse
import os
import pandas as pd
import json
from typing import List
from tqdm import tqdm

from utils.CER import CER, PER


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--result-file",
        type=str,
        required=True,
        help="Json file"
    )
    parser.add_argument(
        "--ref-text-key",
        type=str,
        default='lyric',
        help=""
    )
    parser.add_argument(
        "--pred-text-key",
        type=str,
        default='inference',
        help=""
    )
    parser.add_argument(
        '--ref-timestamp-key',
        type=str,
        default='onset_offset'
    )
    parser.add_argument(
        '--pred-timestamp-key',
        type=str,
        default='inference_onset_offset'
    )

    args = parser.parse_args()
    return args

def is_english(char) -> bool:
    ascii_value = ord(char)
    return (ascii_value >= 65 and ascii_value <= 90) or (ascii_value >= 97 and ascii_value <= 122)

def remove_english(sentence: str):
    result = ''
    for char in sentence:
        if is_english(char) == False:
            result += char
    
    return result


def compute_cer(
    reference: List[str], 
    prediction: List[str],
    is_per: bool=False
):
    metric_name = 'PER' if is_per else 'CER'

    CER_weighted = 0.0
    op_count = {'substitution': 0,
                'insertion': 0,
                'deletion': 0,
                'correct': 0}
    for ref, pred in tqdm(zip(reference, prediction)):
        # Remove All English Characters
        pred = remove_english(pred)

        if is_per:
            cer, nb_map = PER(hypothesis=pred,
                        reference=ref)
        else:
            try:
                cer, nb_map = CER(hypothesis=list(pred),
                                reference=list(ref))
            except:
                cer, nb_map = CER(hypothesis=[],
                                reference=list(ref))
            
        CER_weighted += cer
        op_count['substitution'] += nb_map['S']
        op_count['insertion'] += nb_map['I']
        op_count['deletion'] += nb_map['D']
        op_count['correct'] += nb_map['C']
    
    print('=' * 30)
    print(f"{metric_name}:", CER_weighted / len(reference))
    print("Wrong Operations:")
    for key, value in op_count.items():
        print(f"{key}: {value}")
    print("=" * 30)



def main():
    args = parse_args()

    assert os.path.exists(args.result_file)
    with open(args.result_file, 'r') as f:
        results = json.load(f)

    # CER
    compute_cer(reference=[result[args.ref_text_key] for result in results],
                prediction=[result[args.pred_text_key] for result in results])
    # # PER
    # compute_cer(reference=[result[args.ref_text_key] for result in results],
    #             prediction=[result[args.pred_text_key] for result in results],
    #             is_per=True)
    
if __name__ == "__main__":
    main()