#ENCODED DATASET PREPARATION CODE SAMPLE FOR SPEECH COMMANDS
from datasets import load_dataset, Audio, Dataset
from transformers import EncodecModel, AutoProcessor
import torch
from torch.utils.data import DataLoader
import os
from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio
import soundfile as sf
import pandas  as pd
import librosa
import numpy as np
from tqdm import tqdm
import random
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("path", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

#Get Encodec model version used in AudioGen
model = AudioGen.get_pretrained('facebook/audiogen-medium')
model = model.compression_model

train_set = SubsetSC("training")
test_set = SubsetSC("testing")

labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
def label_to_index(word, labels):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


train_data_embed = []
train_data_embed_y = []
for i in tqdm(range(len(train_set))):
    x_audio = train_set[i][0].reshape(-1)
    audio_len = len(x_audio)
    if audio_len<16000:
        x_audio = torch.nn.ConstantPad1d((0, 16000 - audio_len), 0)(x_audio)
    elif audio_len>16000:
        x_audio = x_audio[:16000]
    x_audio = torch.unsqueeze(x_audio, 0)
    x_audio = torch.unsqueeze(x_audio, 0)
    x_audio = x_audio.to(device="cuda")
    encoder_outputs = model.encode(x_audio)[0].detach()
    train_data_embed.append(encoder_outputs)
    train_data_embed_y.append(torch.tensor(label_to_index(train_set[i][2])))

train_data_embed = torch.cat(train_data_embed)
train_data_embed_y = torch.stack(train_data_embed_y)
torch.save(train_data_embed, 'train_data_embed_SC_deq_x_audiogen.pt')
torch.save(train_data_embed_y, 'train_data_embed_SC_deq_y_audiogen.pt')

valid_data_embed = []
valid_data_embed_y = []
for i in tqdm(range(len(test_set))):
    x_audio = test_set[i][0].reshape(-1)
    audio_len = len(x_audio)
    if audio_len<16000:
        x_audio = torch.nn.ConstantPad1d((0, 16000 - audio_len), 0)(x_audio)
    elif audio_len>16000:
        x_audio = x_audio[:16000]
    x_audio = torch.unsqueeze(x_audio, 0)
    x_audio = torch.unsqueeze(x_audio, 0)
    x_audio = x_audio.to(device="cuda")
    encoder_outputs = model.encode(x_audio)[0].detach()
    valid_data_embed.append(encoder_outputs)
    valid_data_embed_y.append(torch.tensor(label_to_index(test_set[i][2])))

valid_data_embed = torch.cat(valid_data_embed)
valid_data_embed_y = torch.stack(valid_data_embed_y)
torch.save(valid_data_embed, 'valid_data_embed_SC_deq_x_audiogen.pt')
torch.save(valid_data_embed_y, 'valid_data_embed_SC_deq_y_audiogen.pt')

