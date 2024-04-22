import csv
import os, sys
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

if __name__ == "__main__":
    json_path = sys.argv[1]
    output_dict_path = sys.argv[2]

    character_list = []
    pronounce_list = []

    with open(json_path) as json_data:
        data = json.load(json_data)

    for i in range(len(data)):
        for j in range(len(data[i]['aligned_lyrics'])):

            if data[i]['aligned_lyrics'][j][2] not in character_list:
                character_list.append(data[i]['aligned_lyrics'][j][2])

            if '(' in data[i]['aligned_lyrics'][j][3] and '((' not in data[i]['aligned_lyrics'][j][3]:
                cur_pronoun = data[i]['aligned_lyrics'][j][3].split('(')[1].split(')')[0]
            else:
                cur_pronoun = data[i]['aligned_lyrics'][j][3].split('(')[0]

            # Use the first annotation if there are two 
            # (the first is the standard pronounciation based on a disctionary; the second is the actual pronounciation)
            if cur_pronoun not in pronounce_list:
                pronounce_list.append(cur_pronoun)

    pronounce_list.append('<oov>')
    character_list.append('<oov>')

    with open(output_dict_path, 'w') as f:
        json.dump({'pronounce': pronounce_list, 'character': character_list}, f, indent=2, ensure_ascii=False)