from typing import List, Optional
from data_processor.record import Record
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from whisper.tokenizer import Tokenizer
from whisper import log_mel_spectrogram, pad_or_trim
from whisper.audio import N_FRAMES

from data_processor.record import read_data

from utils.audio import load_audio_file

class MultitaskDataset(Dataset):
    def __init__(
        self, 
        records: List[Record],
        pronounce_lookup_table,
        whisper_tokenizer, 
        language: str='zh',
        is_mixture: int=0,
        no_timestamps: bool=True,
        use_ctc: bool=False):

        self.records = records
        self.whisper_tokenizer = whisper_tokenizer

        self.language = language
        self.is_mixture = is_mixture
        self.no_timestamps = no_timestamps

        self.pronounce_lookup_table = pronounce_lookup_table

        self.rarest_dataset_length = min([len(self.records[i]) for i in range(len(self.records))])

        total_batch_count = len(records)
        total_align_batch = 0

    def __len__(self):
        if len(self.records) == 1:
            # For testing / only one set
            return len(self.records[0])
        else:
            # For multiple sets w/ equal weights
            return self.rarest_dataset_length * len(self.records)

    def _get_special_tokens(
            self, 
            is_text_empty: bool, 
            language: str, 
            no_timestamps: bool
    ) -> List[int]:
        if is_text_empty:
            special_tokens = [self.whisper_tokenizer.sot, self.whisper_tokenizer.no_speech]
        else:
            special_tokens = [
                self.whisper_tokenizer.sot,
                self.whisper_tokenizer.special_tokens[f"<|{language}|>"],
                self.whisper_tokenizer.special_tokens["<|transcribe|>"],
            ]
            if no_timestamps:
                special_tokens.append(self.whisper_tokenizer.no_timestamps)

        return special_tokens

    def _encode_text_with_timestamps(
            self,
            text: str,
            lyric_onset_offset: List[List[float]]
    ) -> List[int]:
        
        tokens = []
        for i in range(len(lyric_onset_offset)):
            onset = lyric_onset_offset[i][0]
            offset = lyric_onset_offset[i][1]

            if onset < 0 or onset > 30:
                raise ValueError(f"Invalid timestamp: {onset}")
            if offset < 0 or offset > 30:
                raise ValueError(f"Invalid timestamp: {offset}")

            start_token = self.whisper_tokenizer.timestamp_begin + (onset * 100 // 2)
            end_token = self.whisper_tokenizer.timestamp_begin + (offset * 100 // 2)
            char_token = self.whisper_tokenizer.encode(text[i])

            tokens.append(start_token)
            tokens.extend(char_token)
            tokens.append(end_token)
        
        return tokens

    def _get_transcript_tokens(
            self,
            record: Record,
            no_timestmaps: bool
    ) -> List[int]:
        # print (record.text)
        if no_timestmaps == False:
            text_tokens = self._encode_text_with_timestamps(record.text, record.lyric_onset_offset)
        else:
            text_tokens = self.whisper_tokenizer.encode(''.join(record.text))

        # print (text_tokens)
        return text_tokens

    def _construct_decoder_output(
        self,
        special_tokens: List[int],
        text_tokens: List[int]
    ) -> List[int]:
        decoder_output = special_tokens[1:] + text_tokens + [self.whisper_tokenizer.eot]
        return decoder_output

    
    def __getitem__(self, index):
        if len(self.records) == 1:
            record = self.records[0][index]
        else:
            dataset_index = index // self.rarest_dataset_length
            record_id = int(torch.randint(low=0, high=len(self.records[dataset_index]), size=(1,)))
            record = self.records[dataset_index][record_id]

        # audio = load_audio(record.audio_path, sr=16000)
        audio = load_audio_file(record.audio_path)['speech']

        # Alignment Data
        align_text = record.text

        if record.lyric_onset_offset is not None:     
            lyric_onset_offset = record.lyric_onset_offset
        else:
            lyric_onset_offset = None

        # print (record)
        # Transcription Data
        no_timestamps = self.no_timestamps
        transcript_text_tokens = self._get_transcript_tokens(record, no_timestamps)
        is_text_empty = len(transcript_text_tokens) == 0
        special_tokens = self._get_special_tokens(is_text_empty, self.language, self.no_timestamps)

        decoder_input = special_tokens + transcript_text_tokens
        # print (decoder_input)
        decoder_output = self._construct_decoder_output(special_tokens=special_tokens,
                                                        text_tokens=transcript_text_tokens)

        # print (decoder_output)
        
        return (
            audio,
            align_text,
            lyric_onset_offset,
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(decoder_output, dtype=torch.long),
            record.pronounce
        )

    def get_frame_label(
        self, 
        lyric_tokens, 
        lyric_word_onset_offset,
        output_size_ref,
        hop_size_second: float=0.02
    ):
        fill_value = 0
        # fill_value = -100
        if len(lyric_word_onset_offset) == 0:
            total_frame_num = output_size_ref
        else:
            total_frame_num = int(round(lyric_word_onset_offset[-1][-1] / hop_size_second)) + 1
            
        frame_labels = torch.full((total_frame_num,), fill_value=fill_value)

        for j in range(len(lyric_word_onset_offset)):
            onset_frame = int(round(lyric_word_onset_offset[j][0] / hop_size_second))
            offset_frame = int(round(lyric_word_onset_offset[j][1] / hop_size_second)) + 1
            frame_labels[onset_frame:offset_frame] = lyric_tokens[j]

        return frame_labels

    def collate_fn(self, data):
        audio, align_text, lyric_onset_offset, decoder_input, decoder_output, pronounce = zip(*data)

        for i in range(len(pronounce)):
            for j in range(len(pronounce[i])):
                if '(' in pronounce[i][j] and '((' not in pronounce[i][j]:
                    pronounce[i][j] = pronounce[i][j].split('(')[1].split(')')[0]
                else:
                    pronounce[i][j] = pronounce[i][j].split('(')[0]

        align_text_tokens = []
        for i in range(len(pronounce)):
            if lyric_onset_offset[i] is not None:
                align_text = torch.tensor([self.pronounce_lookup_table[pronounce[i][j]] for j in range(len(pronounce[i]))])
                align_text_tokens.append(align_text)
            else:
                align_text_tokens.append(torch.tensor([-1,]))
        # print (align_text_tokens)
        align_text_tokens = pad_sequence(align_text_tokens, batch_first=True, padding_value=-100)

        frame_labels = []
        for i in range(len(data)):
            # print (align_text_tokens[i], lyric_onset_offset[i], len(audio[i]))
            if lyric_onset_offset[i] is not None:
                # print (audio[i].shape)
                frame_labels.append(self.get_frame_label(align_text_tokens[i], lyric_onset_offset[i], len(audio[i]) // 320))
            else:
                frame_labels.append(None)
        # frame_labels = self.batch_get_frame_label(align_text_tokens, lyric_onset_offset)
        
        # Transcript Token
        decoder_input = pad_sequence(decoder_input, batch_first=True, padding_value=0)
        decoder_output = pad_sequence(decoder_output, batch_first=True, padding_value=-100)

        return audio, align_text_tokens, frame_labels, lyric_onset_offset, decoder_input, decoder_output

def get_multitask_dataloader(
    *data_paths,
    whisper_tokenizer,
    pronounce_lookup_table,
    language: str='zh',
    is_mixture: int=0,
    no_timestamps: bool=True,
    use_ctc: bool=False,
    batch_size: int=1,
    shuffle: bool=False,
) -> DataLoader:
    records = []
    for path in data_paths:
        print (path)
        assert os.path.exists(path)
        records.append(read_data(path))

    dataset = MultitaskDataset(
        records=records,
        pronounce_lookup_table=pronounce_lookup_table,
        whisper_tokenizer=whisper_tokenizer,
        language=language,
        is_mixture=is_mixture,
        no_timestamps=no_timestamps,
        use_ctc=use_ctc
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )