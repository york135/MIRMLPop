import torch
import torch.nn.functional as F
from typing import List

from tqdm import tqdm

# from transformers import AutoTokenizer

import numpy as np
from numba import jit

import sys, time
sys.path.insert(0, '..')
from data_processor.record import read_data

def perform_viterbi(prediction, labels, hop_size_second=0.02):
    log_prediction = F.log_softmax(prediction, dim=2)

    silence_prediction = log_prediction[:,:,0:1]

    log_prediction = torch.clip(log_prediction, min=-1000)[:,:,1:]

    log_silence_prediction = torch.clip(silence_prediction, min=-1000)

    predicted_onset_offset = []

    for i in range(log_prediction.shape[0]):
        cur_label = np.array([labels[i][j] for j in range(len(labels[i])) if labels[i][j] != -100])
        # print (cur_label)
        # blank row: dp_matrix[-1]
        dp_matrix = np.array([[-10000000.0 for k in range(len(cur_label) * 2 + 1)] for j in range(log_prediction.shape[1])])

        backtrace_dp_matrix = np.array([[0 for k in range(len(cur_label) * 2 + 1)] for j in range(log_prediction.shape[1])])

        cur_log_prediction = log_prediction[i].numpy()
        cur_log_silence_prediction = log_silence_prediction[i].numpy()

        dp_matrix[0][0] = cur_log_silence_prediction[0][0]
        dp_matrix[0][1] = cur_log_prediction[0][cur_label[0] - 1]

        # print (time.time())
        dp_matrix, backtrace_dp_matrix = run_viterbi_core(dp_matrix, backtrace_dp_matrix, cur_log_prediction, cur_log_silence_prediction, cur_label)
        # print (time.time())
        if dp_matrix[-1][-1] > dp_matrix[-1][-2]:
            # Go backtrace

            # Get dp_matrix.shape[1] but dp_matrix is not numpy array XD
            correct_path = [len(dp_matrix[0]) - 1, ]
            cur_k = backtrace_dp_matrix[-1][-1]
            # print (backtrace_dp_matrix[-1][-1])
            for j in range(len(dp_matrix)-2, -1, -1):
                correct_path.append(cur_k)
                cur_k = backtrace_dp_matrix[j][cur_k]
        else:

            correct_path = [len(dp_matrix[0]) - 2, ]
            cur_k = backtrace_dp_matrix[-1][-2]
            # print (backtrace_dp_matrix[-1][-2])
            for j in range(len(dp_matrix)-2, -1, -1):
                correct_path.append(cur_k)
                cur_k = backtrace_dp_matrix[j][cur_k]

        correct_path.reverse()

        cur_predicted_onset_offset = []
        cur_pos = 0
        # print (correct_path)
        for k in range(len(cur_label)):
            first_index = correct_path.index(k * 2 + 1)
            last_index = len(correct_path) - correct_path[::-1].index(k * 2 + 1) - 1
            cur_predicted_onset_offset.append([float(first_index) * hop_size_second, float(last_index + 1) * hop_size_second])
        
        predicted_onset_offset.append(list(cur_predicted_onset_offset))
    return predicted_onset_offset

@jit(nopython=True)
def run_viterbi_core(dp_matrix, backtrace_dp_matrix, cur_log_prediction, cur_log_silence_prediction, cur_label):
    # print (cur_label.shape[0] * 2 + 1, log_prediction.shape[1])
    for j in range(1, cur_log_prediction.shape[0]):
        for k in range(cur_label.shape[0] * 2 + 1):
            if k == 0:
                # blank
                backtrace_dp_matrix[j][k] = k
                # dp_matrix[j][k] = dp_matrix[j-1][k] + log_prediction[i][j][0]
                dp_matrix[j][k] = dp_matrix[j-1][k] + cur_log_silence_prediction[j][0]

            elif k == 1:
                if dp_matrix[j-1][k] > dp_matrix[j-1][k-1]:
                    backtrace_dp_matrix[j][k] = k
                    dp_matrix[j][k] = dp_matrix[j-1][k] + cur_log_prediction[j][cur_label[0] - 1]
                else:
                    backtrace_dp_matrix[j][k] = k - 1
                    dp_matrix[j][k] = dp_matrix[j-1][k-1] + cur_log_prediction[j][cur_label[0] - 1]

            elif k % 2 == 0:
                # blank
                if dp_matrix[j-1][k] > dp_matrix[j-1][k-1]:
                    backtrace_dp_matrix[j][k] = k
                    # dp_matrix[j][k] = dp_matrix[j-1][k] + log_prediction[i][j][0]
                    dp_matrix[j][k] = dp_matrix[j-1][k] + cur_log_silence_prediction[j][0]
                else:
                    backtrace_dp_matrix[j][k] = k - 1
                    # dp_matrix[j][k] = dp_matrix[j-1][k-1] + log_prediction[i][j][0]
                    dp_matrix[j][k] = dp_matrix[j-1][k-1] + cur_log_silence_prediction[j][0]

            else:
                if (dp_matrix[j-1][k-2] >= dp_matrix[j-1][k-1] and dp_matrix[j-1][k-2] >= dp_matrix[j-1][k] 
                    and cur_label[k // 2] != cur_label[k // 2 - 1]):
                    # k-2 (last character) -> k
                    backtrace_dp_matrix[j][k] = k - 2
                    dp_matrix[j][k] = dp_matrix[j-1][k-2] + cur_log_prediction[j][cur_label[k // 2] - 1]

                elif dp_matrix[j-1][k] > dp_matrix[j-1][k-1]:
                    # k -> k
                    backtrace_dp_matrix[j][k] = k
                    dp_matrix[j][k] = dp_matrix[j-1][k] + cur_log_prediction[j][cur_label[k // 2] - 1]
                else:
                    # k-1 -> k
                    backtrace_dp_matrix[j][k] = k - 1
                    dp_matrix[j][k] = dp_matrix[j-1][k-1] + cur_log_prediction[j][cur_label[k // 2] - 1]

    return dp_matrix, backtrace_dp_matrix

def get_mae(gt, predict):
    error = 0.0
    cnt = 0
    for i in range(len(gt)):
        for j in range(len(gt[i])):
            error = error + abs(gt[i][j][0] - predict[i][j][0]) + abs(gt[i][j][1] - predict[i][j][1])
            cnt = cnt + 2.0

    error = error / cnt
    return error

def get_mae_v2(gt, predict):
    error = 0.0
    cnt = 0
    for i in range(len(gt)):
        for j in range(min(len(gt[i]), len(predict[i]))):
            error = error + abs(gt[i][j][0] - predict[i][j][0]) + abs(gt[i][j][1] - predict[i][j][1])
            cnt = cnt + 2.0

    error = error / cnt
    return error


def batch_get_frame_label(
        lyric_tokens,
        lyric_word_onset_offset,
        hop_size_second: float=0.02
    ):
        fill_value = -100
        # fill_value = -100

        total_frame_num = max([lyric_word_onset_offset[i][-1][-1] for i in range(len(lyric_word_onset_offset))])
        total_frame_num = int(round(total_frame_num / hop_size_second)) + 1

        frame_labels = torch.full((len(lyric_word_onset_offset), total_frame_num), fill_value=fill_value)

        for i in range(len(lyric_word_onset_offset)):
            for j in range(len(lyric_word_onset_offset[i])):
                onset_frame = int(round(lyric_word_onset_offset[i][j][0] / hop_size_second))
                offset_frame = int(round(lyric_word_onset_offset[i][j][1] / hop_size_second)) + 1
                frame_labels[i][onset_frame: offset_frame] = lyric_tokens[i][j]

        return frame_labels

def get_ce_weight(
    data_path: str,
    tokenizer,
):
    records = read_data(data_path)
    freq = torch.full((len(tokenizer),), 0.001)
    for i in range(len(records)):
        if not hasattr(records[i], "lyric_onset_offset"):
            continue

        target_transcription = [records[i].text]
        labels = tokenizer(target_transcription, 
                           padding=True, 
                           return_tensors="pt").input_ids[:,1:]

        labels[labels == 0] = -100
        labels[labels == 102] = -100

        lyric_word_onset_offset = [records[i].lyric_onset_offset]
        frame_labels = batch_get_frame_label(labels, lyric_word_onset_offset)

        for j in range(len(frame_labels)):
            for k in range(len(frame_labels[j])):
                freq[int(frame_labels[j][k])] += 1

    freq = freq / torch.sum(freq)

    return 1.0 / freq

        