import os
import json
from typing import List, Optional
from dataclasses import dataclass
from ast import literal_eval
import pandas as pd

@dataclass
class Record:
    audio_path: str
    text: str
    lyric_onset_offset: Optional[list]=None

def read_data(
        data_path: str,
    ) -> List[Record]:
    assert os.path.exists(data_path)
    with open(data_path, 'r') as f:
        data_list = json.load(f)

    records = []
    for data in data_list:
        # print (data)
        lyric = ''.join(data['lyric'])
        # print (lyric)
        pronounce = data['pronounce']
        record = Record(audio_path=data['song_path'],
                        text=lyric)

        record.pronounce = list(pronounce)
            
        if 'on_offset' in data:
            record.lyric_onset_offset = data['on_offset']
        # print (record)
        records.append(record)

    return records