import csv
import os, sys
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import statistics
import matplotlib.pyplot as plt

import librosa
from tqdm import tqdm
import soundfile as sf

import argparse

def process_word_label(path):
    rows = np.loadtxt(path, dtype=str, delimiter='\t', ndmin=2)
    # print (rows)
    cur_data = []
    for i in range(len(rows)):
        # print (rows[i])
        # rows[i][0] = float(rows[i][0])
        # rows[i][1] = float(rows[i][1])
        # rows[i][2] = str(rows[i][2])
        no_blank_lyrics = rows[i][2].replace(' ', '')
        cur_data.append([float(rows[i][0]), float(rows[i][1]), no_blank_lyrics[0], no_blank_lyrics[1:]])
    return cur_data


def process_filler_label(path):
    rows = np.loadtxt(path, dtype=str, delimiter='\t', ndmin=2)
    # print (rows)
    cur_data = []
    for i in range(len(rows)):
        # print (rows[i])
        # rows[i][0] = float(rows[i][0])
        # rows[i][1] = float(rows[i][1])
        # rows[i][2] = str(rows[i][2])
        no_blank_lyrics = rows[i][2].replace(' ', '')
        cur_data.append([float(rows[i][0]), float(rows[i][1]), no_blank_lyrics])
    return cur_data

def parse_args():
    parser = argparse.ArgumentParser()
    # Data Argument
    parser.add_argument(
        '-j',
        '--json-label-path',
        type=str,
        required=True
    )
    
    parser.add_argument(
        '-u',
        '--unprocessed-audio-dir',
        type=str,
        required=True
    )

    parser.add_argument(
        '-p',
        '--processed-audio-dir',
        type=str,
        required=True
    )

    parser.add_argument(
        '-train',
        '--train-json-path',
        type=str,
        required=True
    )

    parser.add_argument(
        '-valid',
        '--valid-json-path',
        type=str,
        required=True
    )

    parser.add_argument(
        '-test',
        '--test-json-path',
        type=str,
        required=True
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    json_path = args.json_label_path
    unprocessed_audio_dir = args.unprocessed_audio_dir
    processed_audio_dir = args.processed_audio_dir
    train_json_path = args.train_json_path
    valid_json_path = args.valid_json_path
    test_json_path = args.test_json_path

    if not os.path.exists(processed_audio_dir):
        os.mkdir(processed_audio_dir)

    with open(json_path) as json_data:
        data = json.load(json_data)

    print (len(data))
    # print (data)
    print (unprocessed_audio_dir)
    print (processed_audio_dir)

    training_set = []
    test_set = []
    for i in tqdm(range(len(data))):
        wav_path = os.path.join(unprocessed_audio_dir, data[i]['song_id'] + '.wav')
        if data[i]['data_split'] == 'Train':
            # Split data into at most 30s chunks
            # Use 0.5 second as a threshold to determine 'rest' (backtrace it)
            # If cannot satisfy it, find the closest short pause
            # print (i)

            cur_segment_id = 0

            cur_segment_start_time = data[i]['aligned_lyrics'][0][0]
            cur_segment_start_id = 0
            cur_char_id = 0
            cur_segment = []
            last_rest_point_id = 0
            last_short_pause_point_id = 0
            last_segment_offset = 0
            
            while cur_char_id < len(data[i]['aligned_lyrics']):

                cur_end_time = data[i]['aligned_lyrics'][cur_char_id][1]
                cur_segment_start_time = data[i]['aligned_lyrics'][cur_segment_start_id][0]

                if cur_char_id > 0 and data[i]['aligned_lyrics'][cur_char_id][0] - data[i]['aligned_lyrics'][cur_char_id-1][1] > 0.001:
                    last_short_pause_point_id = cur_char_id - 1

                if cur_char_id > 0 and data[i]['aligned_lyrics'][cur_char_id][0] - data[i]['aligned_lyrics'][cur_char_id-1][1] > 0.5:
                    last_rest_point_id = cur_char_id - 1

                if cur_end_time - cur_segment_start_time >= 30.0:
                    # Now, should find a rest/short pause that is closest to cur_end_time for segmenting.
                    if last_rest_point_id <= cur_segment_start_id:
                        # There is no rest within 30s, so we can only choose last short pause.
                        cur_segment = data[i]['aligned_lyrics'][cur_segment_start_id:last_short_pause_point_id+1]
                        
                        if cur_segment_start_id > 0:
                            # last_char_offset: the offset of the last segment (should start segmenting from this timestamp)
                            last_char_offset = data[i]['aligned_lyrics'][cur_segment_start_id-1][1]
                        else:
                            last_char_offset = 0.0

                        cur_segment_start_id = last_short_pause_point_id + 1
                    else:
                        # Use the rest
                        cur_segment = data[i]['aligned_lyrics'][cur_segment_start_id:last_rest_point_id+1]

                        if cur_segment_start_id > 0:
                            last_char_offset = data[i]['aligned_lyrics'][cur_segment_start_id-1][1]
                        else:
                            last_char_offset = 0.0

                        cur_segment_start_id = last_rest_point_id + 1

                    lyric = [cur_segment[j][2] for j in range(len(cur_segment))]
                    pronounce = [cur_segment[j][3] for j in range(len(cur_segment))]

                    # Include silence (non-vocal) segments that are longer than 5 seconds for training (divide to at most 30s chunks)
                    while cur_segment[0][0] - last_segment_offset > 5.0:
                        silence_duration = min(29.0, cur_segment[0][0] - last_segment_offset)
                        cur_segment_wav, _ = librosa.load(wav_path, sr=44100, offset=last_segment_offset, duration=silence_duration)

                        output_path = os.path.join(processed_audio_dir, data[i]['song_id'] + '_' + str(cur_segment_id) + '.wav')
                        song_id = data[i]['song_id'] + '_' + str(cur_segment_id) + '.wav'

                        sf.write(
                            output_path,
                            cur_segment_wav,
                            44100,
                            "PCM_16",
                        )

                        training_set.append({'song_id': song_id,
                                            'song_path': os.path.abspath(output_path),
                                            'lyric': [],
                                            'pronounce': [],
                                            'on_offset': []})

                        cur_segment_id = cur_segment_id + 1
                        last_segment_offset = last_segment_offset + silence_duration

                    # Include a part of previous silence segment to make the audio duration 30.0s (if possible)
                    duration = cur_segment[-1][1] - cur_segment[0][0]
                    if duration < 30.0:
                        start_time = max(last_char_offset, cur_segment[-1][1] - 30.0)
                    else:
                        start_time = cur_segment[0][0]

                    on_offset = [[cur_segment[j][0] - start_time, cur_segment[j][1] - start_time] for j in range(len(cur_segment))]

                    cur_segment_wav, _ = librosa.load(wav_path, sr=44100, offset=start_time, duration=cur_segment[-1][1] - start_time)
                    output_path = os.path.join(processed_audio_dir, data[i]['song_id'] + '_' + str(cur_segment_id) + '.wav')
                    song_id = data[i]['song_id'] + '_' + str(cur_segment_id) + '.wav'

                    sf.write(
                        output_path,
                        cur_segment_wav,
                        44100,
                        "PCM_16",
                    )

                    training_set.append({'song_id': song_id,
                                        'song_path': os.path.abspath(output_path),
                                        'lyric': lyric,
                                        'pronounce': pronounce,
                                        'on_offset': on_offset})

                    cur_segment_id = cur_segment_id + 1
                    last_segment_offset = cur_segment[-1][1]

                cur_char_id = cur_char_id + 1


            if cur_segment_start_id != len(data[i]['aligned_lyrics']):
                cur_segment = data[i]['aligned_lyrics'][cur_segment_start_id:]

                if cur_segment_start_id > 0:
                    last_char_offset = data[i]['aligned_lyrics'][cur_segment_start_id-1][1]
                else:
                    last_char_offset = 0.0

                duration = cur_segment[-1][1] - cur_segment[0][0]
                if duration < 30.0:
                    start_time = max(last_char_offset, cur_segment[-1][1] - 30.0)
                else:
                    start_time = cur_segment[0][0]

                on_offset = [[cur_segment[j][0] - start_time, cur_segment[j][1] - start_time] for j in range(len(cur_segment))]
                lyric = [cur_segment[j][2] for j in range(len(cur_segment))]
                pronounce = [cur_segment[j][3] for j in range(len(cur_segment))]

                cur_segment_wav, _ = librosa.load(wav_path, sr=44100, offset=start_time, duration=cur_segment[-1][1] - start_time)
                output_path = os.path.join(processed_audio_dir, data[i]['song_id'] + '_' + str(cur_segment_id) + '.wav')
                song_id = data[i]['song_id'] + '_' + str(cur_segment_id) + '.wav'

                sf.write(
                    output_path,
                    cur_segment_wav,
                    44100,
                    "PCM_16",
                )

                training_set.append({'song_id': song_id,
                                    'song_path': os.path.abspath(output_path),
                                    'lyric': lyric,
                                    'pronounce': pronounce,
                                    'on_offset': on_offset})


        elif data[i]['data_split'] == 'Test':
            print ('Test data', wav_path)
            song_id = os.path.basename(wav_path)

            cur_segment = data[i]['aligned_lyrics']

            on_offset = [[cur_segment[j][0], cur_segment[j][1]] for j in range(len(cur_segment))]
            lyric = [cur_segment[j][2] for j in range(len(cur_segment))]
            pronounce = [cur_segment[j][3] for j in range(len(cur_segment))]


            test_set.append({'song_id': song_id,
                                    'song_path': os.path.abspath(wav_path),
                                    'lyric': lyric,
                                    'pronounce': pronounce,
                                    'on_offset': on_offset})
        else:
            print ('It seems like the data_split value is incorrect. Should be either Train or Test.')

    print (len(training_set))
    train_length = int(len(training_set) * 0.9)
    
    with open(train_json_path, 'w') as f:
        json.dump(training_set[:train_length], f, indent=2, ensure_ascii=False)

    with open(valid_json_path, 'w') as f:
        json.dump(training_set[train_length:], f, indent=2, ensure_ascii=False)

    with open(test_json_path, 'w') as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)