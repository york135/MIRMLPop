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

import pypinyin
from pypinyin import lazy_pinyin, Style

from module.align_model import AlignModel
from dataset import get_multitask_dataloader

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
        '--dev-data',
        nargs='+',
        type=str
    )

    parser.add_argument(
        '-d', '--dict-file',
        type=str,
        required=True
    )
    
    # Model Argument
    parser.add_argument(
        '--whisper-model',
        type=str,
        default='medium'
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None
    )

    parser.add_argument(
        '--tokenizer',
        type=str,
        default='bert-base-chinese'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='zh'
    )
    parser.add_argument(
        '--train-alignment',
        action='store_true'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda'
    )

    # Training Argument
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=1
    )
    parser.add_argument(
        '--dev-batch-size',
        type=int,
        default=1
    )
    parser.add_argument(
        '--accum-grad-steps',
        type=int,
        default=16
    )
    parser.add_argument(
        '--freeze-encoder',
        action='store_true'
    )
    parser.add_argument(
        '--use-ctc-loss',
        action='store_true'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3
    )
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--train-steps',
        type=int,
        default=4000
    )
    parser.add_argument(
        '--eval-steps',
        type=int,
        default=200
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=200
    )

    parser.add_argument(
        '--save-dir',
        type=str,
        default='result'
    )
    parser.add_argument(
        '--save-all-checkpoints',
        type=bool,
        default=True
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

def save_args(args: argparse.Namespace, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(vars(args), indent=4, ensure_ascii=False))

def infinite_iter(data_loader: DataLoader) -> Iterator:
    while True:
        for batch in data_loader:
            yield batch

# >>> Batch handling >>>
def rebatch_handler(
    batch,
    is_multitask: bool=False
):
    audios = [item[0] for item in batch]
    align_text = torch.stack([item[1] for item in batch])

    if is_multitask:
        frame_labels = pad_sequence([item[2] for item in batch],
                                     batch_first=True,
                                     padding_value=-100)
        lyric_onset_offset = [item[3] for item in batch]
    else:
        frame_labels = None
        lyric_onset_offset = None
    
    decoder_input = torch.stack([item[4] for item in batch])
    decoder_output = torch.stack([item[5] for item in batch])

    return audios, align_text, frame_labels, lyric_onset_offset, decoder_input, decoder_output

def split_batch(batch):
    # multitask batch => alignment + decoder transcript
    # transcript batch => decoder transcript
    multitask_batch = []
    transcript_batch = []
    
    # base on if frame labels exists
    # frame_labels = item[2] 
    for item in zip(*batch):
        if item[2] is not None:
            multitask_batch.append(item)
        else:
            transcript_batch.append(item)

    if len(multitask_batch) > 0:
        multitask_batch = rebatch_handler(multitask_batch, is_multitask=True)
    else:
        multitask_batch = None
    if len(transcript_batch) > 0:
        transcript_batch = rebatch_handler(transcript_batch, is_multitask=False)
    else:
        transcript_batch = None

    return multitask_batch, transcript_batch

def get_align_batch(batch):
    align_batch = []

    # base on if frame labels exists
    # frame_labels = item[2] 
    for item in zip(*batch):
        if item[2] is not None:
            align_batch.append(item)

    if len(align_batch) > 0:
        align_batch = rebatch_handler(align_batch, is_multitask=True)
    else:
        align_batch = None
    
    return align_batch

# <<< Batch handling <<<


def train_step(
    pronounce_lookup_table,
    model: AlignModel,
    train_iter: Iterator,
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    accum_grad_steps: int,
    max_grad_norm: float,
    loss_fn: dict,
    vocab_size: int,
    use_ctc_loss: bool=False,
    get_orig_len: bool=True,
    allow_transcript: bool=True,
):
    model.train()
    
    device = model.device

    losses = {'total': 0,
            'align_ce': 0,
            'align_ctc': 0,
            'trans_ce': 0,
            'trans_ctc': 0}

    for _ in range(accum_grad_steps):
        batch = next(train_iter)
        multitask_batch, transcript_batch = split_batch(batch)
        # align_batch = get_align_batch(batch)

        # audios
        # y_text
        # frame_labels
        # lyric_word_onset_offset
        # decoder_input
        # decoder_output
        if multitask_batch is not None:
            decoder_input = multitask_batch[4].to(device) if allow_transcript else None

            multi_align_logits, multi_trans_logits = model.frame_manual_forward(
                multitask_batch[0],
                decoder_input,
                get_orig_len=get_orig_len
            )
            if model.train_alignment:
                # print (multitask_batch[2])
                # print (multitask_batch[1])
                # for i in range(len(multitask_batch[1])):
                #     for j in range(len(multitask_batch[1][i])):
                #         if multitask_batch[1][i][j] != -100:
                #             multitask_batch[1][i][j] = pronounce_lookup_table[multitask_batch[1][i][j]]


                # for i in range(len(multitask_batch[2])):
                #     for j in range(len(multitask_batch[2][i])):
                #         if multitask_batch[2][i][j] != -100:
                #             multitask_batch[2][i][j] = pronounce_lookup_table[multitask_batch[2][i][j]]

                multi_align_ce_loss = compute_ce_loss(multi_align_logits,
                                                      multitask_batch[2].to(device),
                                                      vocab_size=vocab_size,
                                                      loss_fn=loss_fn,
                                                      compute_sil=use_ctc_loss,
                                                      device=device)
                if use_ctc_loss:
                    # print (multitask_batch[1])
                    multi_align_ctc_loss = compute_ctc_loss(multi_align_logits[:, :, : vocab_size],
                                                            multitask_batch[1].to(device),
                                                            device=device)
            else:
                multi_align_ce_loss = torch.tensor(0, device=device, dtype=torch.float)
                multi_align_ctc_loss = torch.tensor(0, device=device, dtype=torch.float)
            
            if model.train_transcript:
                multi_trans_loss = F.cross_entropy(multi_trans_logits.permute(0, 2, 1), multitask_batch[-1].to(device))
            else:
                multi_trans_loss = torch.tensor(0, device=device, dtype=torch.float)

            # multitask_loss = (multi_align_ce_loss * 2) + multi_trans_loss
            multitask_loss = multi_align_ce_loss + multi_trans_loss
            if use_ctc_loss:
                # multitask_loss += (multi_align_ctc_loss * 2)
                multitask_loss += multi_align_ctc_loss
        else:
            multi_align_ce_loss = torch.tensor(0, device=device, dtype=torch.float)
            multi_align_ctc_loss = torch.tensor(0, device=device, dtype=torch.float)
            multi_trans_loss = torch.tensor(0, device=device, dtype=torch.float)
            multitask_loss = torch.tensor(0, device=device, dtype=torch.float)

        if transcript_batch is not None and allow_transcript:
            trans_align_logits, trans_logits = model.frame_manual_forward(
                transcript_batch[0],
                transcript_batch[4].to(device),
                get_orig_len=get_orig_len
            )

            if model.train_transcript:
                transcript_loss = F.cross_entropy(trans_logits.permute(0, 2, 1), transcript_batch[-1].to(device))
            else:
                transcript_loss = torch.tensor(0, device=device, dtype=torch.float)

            # if use_ctc_loss and model.train_alignment:
            if False:
                # for i in range(len(transcript_batch[1])):
                #     for j in range(len(transcript_batch[1][i])):
                #         if transcript_batch[1][i][j] != -100:
                #             transcript_batch[1][i][j] = pronounce_lookup_table[transcript_batch[1][i][j]]
                # print (transcript_batch[1][i][j])

                transcript_ctc_loss = compute_ctc_loss(trans_align_logits[:, :, : vocab_size], 
                                                       transcript_batch[1].to(device),
                                                       device=device)
            else:
                transcript_ctc_loss = torch.tensor(0, device=device, dtype=torch.float)
            
            transcript_loss += transcript_ctc_loss
        else:
            transcript_loss = torch.tensor(0, device=device, dtype=torch.float)
            transcript_ctc_loss = torch.tensor(0, device=device, dtype=torch.float)
            

        loss = (multitask_loss + transcript_loss) / accum_grad_steps
        loss.backward()

        losses['total'] += loss.item()
        # losses['align_ce'] += (multi_align_ce_loss.item() * 2) / accum_grad_steps
        losses['align_ce'] += multi_align_ce_loss.item() / accum_grad_steps
        losses['trans_ce'] += (multi_trans_loss.item() + transcript_loss.item() - transcript_ctc_loss.item()) / accum_grad_steps
        if use_ctc_loss:
            # losses['align_ctc'] += (multi_align_ctc_loss.item() * 2) / accum_grad_steps
            losses['align_ctc'] += multi_align_ctc_loss.item() / accum_grad_steps
            losses['trans_ctc'] += transcript_ctc_loss.item() / accum_grad_steps

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return losses


@torch.no_grad()
def evaluate(
    pronounce_lookup_table,
    model: AlignModel,
    dev_loader: DataLoader,
    loss_fn: dict,
    vocab_size: int,
    use_ctc_loss: bool=False,
    get_orig_len: bool=True
):
    device = model.device

    model.eval()
    losses = {'total': 0,
            'align_ce': 0,
            'align_ctc': 0,
            'trans_ce': 0,
            'trans_ctc': 0}

    # mel, y_text, frame_labels, lyric_word_onset_offset, decoder_input, decoder_output
    for batch in tqdm(dev_loader):
        # print (batch)
        multitask_batch, transcript_batch = split_batch(batch)

        # mel
        # y_text
        # frame_labels
        # lyric_word_onset_offset
        # decoder_input
        # decoder_output
        if multitask_batch is not None:
            # print (multitask_batch[0])
            # print (multitask_batch[4])

            multi_align_logits, multi_trans_logits = model.frame_manual_forward(
                multitask_batch[0],
                multitask_batch[4].to(device),
                get_orig_len=get_orig_len
            )
            # print (multi_align_logits)

            if model.train_alignment:
                # print (multitask_batch[2])

                # for i in range(len(multitask_batch[1])):
                #     for j in range(len(multitask_batch[1][i])):
                #         if multitask_batch[1][i][j] != -100:
                #             # print (multitask_batch[1][i][j], token_pinyin[multitask_batch[1][i][j]])
                #             # print (pinyin_lookup_table[token_pinyin[multitask_batch[1][i][j]]])
                #             multitask_batch[1][i][j] = pronounce_lookup_table[multitask_batch[1][i][j]]

                # for i in range(len(multitask_batch[2])):
                #     for j in range(len(multitask_batch[2][i])):
                #         if multitask_batch[2][i][j] != -100:
                #             multitask_batch[2][i][j] = pronounce_lookup_table[multitask_batch[2][i][j]]
                # print (multi_align_logits)
                multi_align_ce_loss = compute_ce_loss(multi_align_logits,
                                                    multitask_batch[2].to(device),
                                                    vocab_size=vocab_size,
                                                    loss_fn=loss_fn,
                                                    compute_sil=use_ctc_loss,
                                                    device=device)
                # print (multi_align_ce_loss)
                # print (multitask_batch[2])
                # print (multitask_batch[1])
                if use_ctc_loss:
                    multi_align_ctc_loss = compute_ctc_loss(multi_align_logits[:, :, : vocab_size],
                                                            multitask_batch[1].to(device),
                                                            device=device)
            else:
                multi_align_ce_loss = torch.tensor(0, device=device, dtype=torch.float)
                multi_align_ctc_loss = torch.tensor(0, device=device, dtype=torch.float)

            if model.train_transcript:
                multi_trans_loss = F.cross_entropy(multi_trans_logits.permute(0, 2, 1), multitask_batch[-1].to(device))
            else:
                multi_trans_loss = torch.tensor(0, device=device, dtype=torch.float)

            multitask_loss = multi_align_ce_loss + multi_trans_loss
            if use_ctc_loss:
                multitask_loss += multi_align_ctc_loss
        else:
            multi_align_ce_loss = torch.tensor(0, device=device, dtype=torch.float)
            multi_align_ctc_loss = torch.tensor(0, device=device, dtype=torch.float)
            multi_trans_loss = torch.tensor(0, device=device, dtype=torch.float)
            multitask_loss = torch.tensor(0, device=device, dtype=torch.float)

        if transcript_batch is not None:
            trans_align_logits, trans_logits = model.frame_manual_forward(
                transcript_batch[0],
                transcript_batch[4].to(device),
                get_orig_len=get_orig_len
            )

            if model.train_transcript:
                transcript_loss = F.cross_entropy(trans_logits.permute(0, 2, 1), transcript_batch[-1].to(device))
            else:
                transcript_loss = torch.tensor(0, device=device, dtype=torch.float)

            # if use_ctc_loss and model.train_alignment:
            if False:

                # for i in range(len(transcript_batch[1])):
                #     for j in range(len(transcript_batch[1][i])):
                #         if transcript_batch[1][i][j] != -100:
                #             transcript_batch[1][i][j] = pronounce_lookup_table[transcript_batch[1][i][j]]
                # print (transcript_batch[1][i][j])

                trans_ctc_loss = compute_ctc_loss(trans_align_logits[:, :, : vocab_size],
                                                  transcript_batch[1].to(device),
                                                  device=device)
            else:
                trans_ctc_loss = torch.tensor(0, device=device, dtype=torch.float)   
                
            transcript_loss += trans_ctc_loss

        else:
            transcript_loss = torch.tensor(0, device=device, dtype=torch.float)
            trans_ctc_loss = torch.tensor(0, device=device, dtype=torch.float)


        losses['total'] += multitask_loss.item() + transcript_loss.item()
        losses['align_ce'] += multi_align_ce_loss.item()
        losses['trans_ce'] += (multi_trans_loss.item() + transcript_loss.item() - trans_ctc_loss.item())
        if use_ctc_loss:
            losses['align_ctc'] += multi_align_ctc_loss.item()
            losses['trans_ctc'] += trans_ctc_loss.item()
        
    for key in losses.keys():
        losses[key] /= len(dev_loader)

    return losses


def save_model(model, save_path: str) -> None:
    # save model in half precision to save space
    #model = copy.deepcopy(model).half()
    # save model weights and config in a dictionary that can be loaded with `whisper.load_model`
    torch.save(model.state_dict(), save_path)

def main_loop(
    pronounce_lookup_table,
    model,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    loss_fn: dict,
    args: argparse.Namespace,
    get_orig_len: bool=True,
) -> None:
    init_losses = evaluate(pronounce_lookup_table, model, 
                           dev_loader,
                           vocab_size=len(pronounce_lookup_table) + 1, 
                           loss_fn=loss_fn,
                           use_ctc_loss=args.use_ctc_loss,
                           get_orig_len=get_orig_len)
    
    min_loss = init_losses['total']
    min_align_loss = init_losses['align_ce'] + init_losses['align_ctc']
    min_trans_loss = init_losses['trans_ce']
    min_trans_ctc_loss = init_losses['trans_ctc']
    
    avg_losses = {'total': 0,
            'align_ce': 0,
            'align_ctc': 0,
            'trans_ce': 0,
            'trans_ctc': 0}

    # Force Terminate if no_improve_count >= 5
    # no_improve_count = 0

    if args.use_ctc_loss:
        print(f"Initial loss: {min_loss}, Align CE loss: {init_losses['align_ce']}, Align CTC loss: {init_losses['align_ctc']}\
            , Transcript loss: {init_losses['trans_ce']}, Transcript CTC loss: {init_losses['trans_ctc']}")
    else:
        print(f"Initial loss: {min_loss}, Align loss: {init_losses['align_ce']}, Transcript loss: {init_losses['trans_ce']}")
    
    transcript_late_start_steps = 0
    print(f'decoder finetune delayed until step {transcript_late_start_steps}')
    pbar = tqdm(range(1, args.train_steps + 1))
    train_iter = infinite_iter(train_loader)
    for step in pbar:
        allow_transcript = True if step >= transcript_late_start_steps else False

        train_losses = train_step(
            pronounce_lookup_table, model,
            train_iter,
            optimizer,
            scheduler,
            args.accum_grad_steps,
            args.max_grad_norm,
            vocab_size=len(pronounce_lookup_table) + 1,
            loss_fn=loss_fn,
            use_ctc_loss=args.use_ctc_loss,
            get_orig_len=get_orig_len,
            allow_transcript=allow_transcript
        )
        if args.use_ctc_loss:
            pbar.set_postfix({
                "loss": train_losses['total'],
                "align_ce_loss": train_losses['align_ce'],
                "align_ctc_loss": train_losses['align_ctc'],
                "transcript_loss": train_losses['trans_ce'],
                "trans_ctc_loss": train_losses['trans_ctc']
            })
        else:
            pbar.set_postfix({
                "loss": train_losses['total'],
                "align_ce_loss": train_losses['align_ce'],
                "transcript_loss": train_losses['trans_ce']
            })

        for key in avg_losses.keys():
            avg_losses[key] += train_losses[key]

        if step % args.eval_steps == 0:
            eval_losses = evaluate(
                pronounce_lookup_table, model, 
                dev_loader, 
                vocab_size= len(pronounce_lookup_table) + 1,
                loss_fn=loss_fn,
                use_ctc_loss=args.use_ctc_loss,
                get_orig_len=get_orig_len
            )

            if args.use_ctc_loss:
                tqdm.write(f"Step {step}: valid loss={eval_losses['total']}, valid align CE loss={eval_losses['align_ce']}, valid align CTC loss={eval_losses['align_ctc']}, valid transcript loss={eval_losses['trans_ce']}, valid transcript ctc loss={eval_losses['trans_ctc']}")
                tqdm.write(f"Step {step}: train loss={avg_losses['total'] / args.eval_steps}, \
                    train align CE loss={avg_losses['align_ce'] / args.eval_steps}, train align CTC loss={avg_losses['align_ctc'] / args.eval_steps}, \
                    train transcript loss={avg_losses['trans_ce'] / args.eval_steps}, train transcript ctc loss={avg_losses['trans_ctc'] / args.eval_steps}")
            else:
                tqdm.write(f"Step {step}: valid loss={eval_losses['total']}, valid align CE loss={eval_losses['align_ce']}, valid transcript loss={eval_losses['trans_ce']}")
                tqdm.write(f"Step {step}: train loss={avg_losses['total'] / args.eval_steps}, train align CE loss={avg_losses['align_ce'] / args.eval_steps}, train transcript loss={avg_losses['trans_ce'] / args.eval_steps}")

            # Reset Average Loss
            for key in avg_losses.keys():
                avg_losses[key] = 0

            if eval_losses['total'] < min_loss:
                min_loss = eval_losses['total']
                tqdm.write("Saving The Best Model")
                save_model(model, f"{args.save_dir}/best_model.pt")

            if (eval_losses['align_ce'] + eval_losses['align_ctc']) < min_align_loss:
                min_align_loss = eval_losses['align_ce'] + eval_losses['align_ctc']
                tqdm.write("Saving The Best Align Model")
                save_model(model, f"{args.save_dir}/best_align_model.pt")

            if eval_losses['trans_ce'] < min_trans_loss:
                min_trans_loss = eval_losses['trans_ce']
                tqdm.write("Saving The Best Transcript Model")
                save_model(model, f"{args.save_dir}/best_trans_model.pt")

            if args.save_all_checkpoints:
                save_model(model, f"{args.save_dir}/step{step}.pt")

            save_model(model, f"{args.save_dir}/last_model.pt")

def compute_ce_loss(
    logits: torch.Tensor, 
    frame_labels: torch.Tensor, 
    loss_fn: dict,
    vocab_size: int,
    compute_sil: bool=False,
    device: str='cuda'
) -> torch.Tensor:
    fill_value = -100 if compute_sil else 0

    frame_labels = frame_labels[:, : logits.shape[1]]
    if frame_labels.shape[1] < logits.shape[1]:
        frame_labels = torch.cat((frame_labels, 
                                  torch.full((frame_labels.shape[0], logits.shape[1] - frame_labels.shape[1]), 
                                             fill_value=-100, 
                                             device=device)), 
                                  dim=1)

    if compute_sil == False:    
        loss = F.cross_entropy(logits.permute(0, 2, 1), frame_labels)
        return loss

    frame_labels[frame_labels != -100] -= 1

    word_ce_loss = loss_fn['ce_loss'](logits[:, :, 1: vocab_size].transpose(1, 2), frame_labels)
    word_ce_loss = torch.nan_to_num(word_ce_loss)

    silence_label = torch.where(frame_labels == -100, 1, 0)
    silence_ce_loss = loss_fn['silence_ce_loss'](logits[:, :, vocab_size], silence_label.float())
    silence_ce_loss = torch.nan_to_num(silence_ce_loss)

    return word_ce_loss + silence_ce_loss

def compute_ctc_loss(
    logits,
    labels,
    device: str='cuda',
):
    output_log_sm = F.log_softmax(logits, dim=2)
    output_log_sm = output_log_sm.transpose(0, 1)

    # print (output_log_sm.shape, labels.shape)
    # print (labels)

    input_lengths = torch.full(size=(output_log_sm.shape[1],), 
                               fill_value=output_log_sm.shape[0], 
                               dtype=torch.long).to(device)
    target_length = torch.sum(labels != -100, dim=1)
    # print (target_length)
    # print (input_lengths)
    # print (labels)

    cur_ctc_loss = F.ctc_loss(output_log_sm, labels, input_lengths, target_length)
    return cur_ctc_loss

def load_align_model_and_tokenizer(
    model_path: str,
    train_args,
    model_args,
    device: str='cuda'
) -> AlignModel:
    assert os.path.exists(model_path)
    whisper_model_name = train_args.whisper_model

    whisper_model = whisper.load_model(whisper_model_name, device=device)

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
    
    model.to(device)
    return model



def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_args(args, f"{args.save_dir}/args.json")


    device = args.device
    if 'cuda' in device and torch.cuda.is_available() == False:
        device = 'cpu'

    pronounce_lookup_table = {}
    with open(args.dict_file) as json_data:
        dictionary = json.load(json_data)

    # with open('/work/d10922010/lyrics_recog/mandarin_dictionary.json') as json_data:
    #     dictionary = json.load(json_data)

    for i in range(len(dictionary['pronounce'])):
        pronounce_lookup_table[dictionary['pronounce'][i]] = i + 1

    # print (pronounce_lookup_table)

    model_args = {'embed_dim': WHISPER_DIM[args.whisper_model],
                  'hidden_dim': 384,
                  'output_dim': len(pronounce_lookup_table) + 1 + args.use_ctc_loss,
                  'bidirectional': True,
                  'freeze_encoder': args.freeze_encoder,
                  'train_alignment': args.train_alignment,
                  'train_transcript': True,}

    print(model_args)


    whisper_tokenizer = get_tokenizer(multilingual=".en" not in args.whisper_model, task="transcribe")

    if args.model_path is not None:
        multitask_model = load_align_model_and_tokenizer(args.model_path, args, model_args, device=device)
        multitask_model.train_alignment = model_args['train_alignment']
        multitask_model.train_transcript = model_args['train_transcript']
    else:
        whisper_model = whisper.load_model(args.whisper_model, device=device)

        multitask_model = AlignModel(
            whisper_model=whisper_model,
            embed_dim=model_args['embed_dim'],
            hidden_dim=model_args['hidden_dim'],
            output_dim=model_args['output_dim'],
            bidirectional=model_args['bidirectional'],
            freeze_encoder=model_args['freeze_encoder'],
            train_alignment=model_args['train_alignment'],
            train_transcript=model_args['train_transcript'],
            device=device
        ).to(device)


    with open(f"{args.save_dir}/model_args.json", 'w') as f:
        json.dump(model_args, f, indent=4)
    
    optimizer = torch.optim.AdamW([
                                    {'params': multitask_model.align_rnn.parameters(), 'lr': args.lr,},
                                    {'params': multitask_model.whisper_model.encoder.parameters(), 'lr': args.lr / 100,},
                                    {'params': multitask_model.whisper_model.decoder.parameters(), 'lr': args.lr / 100,}],
                                    lr=args.lr,
                                    weight_decay=1e-2)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.train_steps
    )

    # assert os.path.exists(args.train_data)
    train_dataloader = get_multitask_dataloader(
        *args.train_data,
        pronounce_lookup_table=pronounce_lookup_table,
        whisper_tokenizer=whisper_tokenizer,
        language=args.language,
        no_timestamps=True,
        use_ctc=args.use_ctc_loss,
        batch_size=args.train_batch_size,
        shuffle=True
    )

    dev_dataloader = get_multitask_dataloader(
        *args.dev_data,
        pronounce_lookup_table=pronounce_lookup_table,
        whisper_tokenizer=whisper_tokenizer,
        language=args.language,
        no_timestamps=True,
        use_ctc=args.use_ctc_loss,
        batch_size=args.dev_batch_size,
        shuffle=False
    )


    loss_fn = {'ce_loss': nn.CrossEntropyLoss(),
               'silence_ce_loss': nn.BCEWithLogitsLoss()}

    main_loop(pronounce_lookup_table,
        model=multitask_model,
        train_loader=train_dataloader,
        dev_loader=dev_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        args=args,
        get_orig_len=False
    )


if __name__ == "__main__":
    main()