import numpy as np
import itertools
from typing import List

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import whisper
from whisper.audio import N_FRAMES, pad_or_trim, log_mel_spectrogram

from torchaudio.transforms import FrequencyMasking, TimeMasking

class Transformer(nn.Module):
    pass
   
class RNN(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size,
        output_size,
        num_layers: int=2,
        dropout: float=0.1,
        batch_first: bool=True, 
        bidirectional: bool=True,
    ) -> None:
        super().__init__()
        # self.pre_fc = nn.Linear(input_size, hidden_size)
        # self.rnn = nn.GRU(input_size=hidden_size,
        #                     hidden_size=hidden_size,
        #                     num_layers=num_layers,
        #                     dropout=dropout,
        #                     batch_first=batch_first,
        #                     bidirectional=bidirectional)

        self.rnn = nn.GRU(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=batch_first,
                            bidirectional=bidirectional)
        
        # self.ln = nn.LayerNorm(hidden_size + (bidirectional * hidden_size))

        self.activate = nn.Mish()

        self.fc = nn.Linear(hidden_size + (bidirectional * hidden_size), 
                            output_size)

    def forward(self, x):
        # x = self.pre_fc(x)
        out, _ = self.rnn(x)
        # out = self.ln(out)
        out = self.activate(out)
        out = self.fc(out)

        return out

class AlignModel(torch.nn.Module):
    def __init__(self,
        whisper_model: whisper.Whisper,
        embed_dim: int=1280,
        hidden_dim: int=384,
        dropout: float=0.15,
        output_dim: int=10000,
        bidirectional: bool=True,
        freeze_encoder: bool=False,
        train_alignment: bool=True,
        train_transcript: bool=False,
        device: str='cuda'
        ) -> None:
        super().__init__()
        self.whisper_model = whisper_model

        # self.time_masking = TimeMasking(time_mask_param=100)
        self.freq_masking = FrequencyMasking(freq_mask_param=27)
        
        # print (whisper_model)

        # Text Alignment
        self.align_rnn = RNN(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            output_size=output_dim,
                            bidirectional=bidirectional,
                            dropout=dropout)

        self.freeze_encoder = freeze_encoder

        self.train_alignment = train_alignment
        self.train_transcript = train_transcript

        self.device = device

    def frame_manual_forward(
        self,
        audios: List[np.ndarray],
        y_in=None,
        get_orig_len: bool=True
    ):
        audios = np.stack((itertools.zip_longest(*audios, fillvalue=0)), axis=1).astype('float32')
        mel = log_mel_spectrogram(audios).to(self.device)

        # print (mel.shape)
        # mel = self.time_masking(mel, mask_value=mel.mean())
        # if self.training == True:
        #     mel = self.freq_masking(mel, mask_value=mel.mean())

        align_logit = None
        if get_orig_len:
            if mel.shape[-1] <= N_FRAMES:
                orig_mel_len = int(round(mel.shape[-1] / 2.0))
                mel = pad_or_trim(mel, N_FRAMES)
                
                embed_pad = self.whisper_model.embed_audio(mel)
                embed = embed_pad[:, : orig_mel_len, :]
            else:
                embed = []
                for i in range(0, mel.shape[-1], N_FRAMES):
                    start = i
                    end = min(i + N_FRAMES, mel.shape[-1])
                    orig_mel_len = int(round((end - start) / 2.0))

                    cur_mel = pad_or_trim(mel[:, :, start: end], N_FRAMES)
                    cur_embed = self.whisper_model.embed_audio(cur_mel)

                    embed.append(cur_embed[:, : orig_mel_len, :])
                embed = torch.cat(embed, dim=1)
                embed_pad = embed[:, : (N_FRAMES // 2), :]
            if self.train_alignment:
                align_logit = self.align_rnn(embed)
        else:
            mel = pad_or_trim(mel, N_FRAMES)
            mel = pad_sequence(mel, batch_first=True, padding_value=0)
            embed_pad = self.whisper_model.embed_audio(mel)
            
            if self.train_alignment:
                align_logit = self.align_rnn(embed_pad)
        

        transcribe_logit = None
        if self.train_transcript and y_in is not None:
            transcribe_logit = self.whisper_model.logits(tokens=y_in,
                                                         audio_features=embed_pad)

        return align_logit, transcribe_logit


    def forward(
        self,
        mel,
        y_in=None):
        # x => Mel
        # y_in => whisper decoder input
        # You can ignore y_in if you are doing alignment task
        
        
        if self.freeze_encoder:
            with torch.no_grad():
                embed = self.whisper_model.embed_audio(mel)
        else:
            embed = self.whisper_model.embed_audio(mel)

        # Align Logit
        align_logit = None
        if self.train_alignment:
            align_logit = self.align_rnn(embed)

        # Transcribe Logit
        transcribe_logit = None
        if self.train_transcript and y_in is not None:
            transcribe_logit = self.whisper_model.logits(tokens=y_in,
                                                         audio_features=embed)

        return align_logit, transcribe_logit