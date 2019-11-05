import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from models.ops import Flatten, CNN_Permute, RNN_Permute
from src.data_manager import Vocab
import numpy as np

class MultiNet(nn.Module):
    def __init__(self, hparams ,vocab: Vocab, modal='multi'):
        super().__init__()
        self.modal = modal
        if modal=='audio':
            self.audionet = AudioNet(hparams)
        elif modal=='lyrics':
            self.lyricsnet = LyricsNet(hparams, vocab)
        elif modal=='multi':
            self.audionet = AudioNet(hparams)
            self.lyricsnet = LyricsNet(hparams, vocab)
        else:
            raise ValueError('Modal shoud be specified.')

        if hparams.label_name == 'arousal' or hparams.label_name == 'valence':
            self.num_output = 1
        elif hparams.label_name == 'Label_kMean_4' or hparams.label_name == 'Label_abs_4':
            self.num_output = 4
        elif hparams.label_name == 'Label_kMean_16' or hparams.label_name == 'Label_kMean_16':
            self.num_output = 16
        else:
            self.num_output = 64

        self._classifier = nn.Sequential(nn.Linear(in_features=1456, out_features=100),
                                        # nn.Tanh(),
                                        nn.Hardtanh(min_val=-0.5, max_val=0.5),
                                        nn.Dropout(),
                                        nn.Linear(in_features=100, out_features=self.num_output))        
        # self.init_weights()

    def forward(self, audio, lyrics):
        if self.modal=='audio':
            score, _ = self.audionet(audio)
        elif self.modal=='lyrics':
            score, _ = self.lyricsnet(lyrics)
        elif self.modal=='multi':
            _, audioflat = self.audionet(audio)
            _, lyricsflat = self.lyricsnet(lyrics)
            concat = torch.cat((audioflat, lyricsflat),dim=1)
            score = self._classifier(concat)        
        return score
            
class AudioNet(nn.Module):
    def __init__(self, hparams):
        super(AudioNet, self).__init__()

        if hparams.label_name == 'arousal' or hparams.label_name == 'valence':
            self.num_output = 1
        elif hparams.label_name == 'Label_kMean_4' or hparams.label_name == 'Label_abs_4':
            self.num_output = 4
        elif hparams.label_name == 'Label_kMean_16' or hparams.label_name == 'Label_kMean_16':
            self.num_output = 16
        else:
            self.num_output = 64

        self.conv0 = nn.Sequential(
			nn.Conv1d(hparams.num_mels, 32, kernel_size=8, stride=1, padding=0),
            nn.MaxPool1d(4, stride=4),
			nn.BatchNorm1d(32)
			)

        self.conv1 = nn.Sequential(
			nn.Conv1d(32, 16, kernel_size=8, stride=1, padding=0),
            nn.MaxPool1d(4, stride=4),
			nn.BatchNorm1d(16),
			)

        self._classifier = nn.Sequential(nn.Linear(in_features=976, out_features=64),
                                        nn.Tanh(),
                                        # nn.Hardtanh(min_val=-0.5, max_val=0.5),
                                        nn.Dropout(),
                                        nn.Linear(in_features=64, out_features=self.num_output),
                                        )

        self.apply(self._init_weights)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv0(x)
        x = self.conv1(x)
        flatten = x.view(x.size(0), -1)
        score = self._classifier(flatten)
        return score, flatten
        
    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)

        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

class LyricsNet(nn.Module):
    def __init__(self, hparams, vocab: Vocab):
        super(LyricsNet, self).__init__()

        if hparams.label_name == 'arousal' or hparams.label_name == 'valence':
            self.num_output = 1
        elif hparams.label_name == 'Label_kMean_4' or hparams.label_name == 'Label_abs_4':
            self.num_output = 4
        elif hparams.label_name == 'Label_kMean_16' or hparams.label_name == 'Label_kMean_16':
            self.num_output = 16
        else:
            self.num_output = 64
            
        self._lstm_hid_dim = hparams.lstm_hid_dim
        self._batch_size = hparams.batch_size

        self._embedding =nn.Embedding(len(vocab), hparams.emb_dim, vocab.to_indices(vocab.padding_token))
        self._cnn_permute = CNN_Permute()
        self._conv0 = nn.Sequential(nn.Conv1d(in_channels=hparams.emb_dim, out_channels=16, kernel_size=2, stride=2),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2)
                                    )
        self._flatten = Flatten()
        self._rnn_permute = RNN_Permute()
        self._lstm = nn.LSTM(16, hparams.lstm_hid_dim, batch_first=False)
        self.hidden_state = self.init_hidden()

        self._classifier = nn.Sequential(nn.Linear(in_features=480, out_features=64),
                                         nn.Tanh(),
                                        #  nn.Hardtanh(min_val=-0.5, max_val=0.5),
                                         nn.Dropout(0.5),
                                         nn.Linear(in_features=64, out_features=self.num_output))
        self.apply(self._initailze)
        
    
    def forward(self, x):
        x = self._embedding(x)      # ([32, 50, 100]) Batch, seq_len, Input_D
        x = self._cnn_permute(x)        # ([32, 100, 50]) Batch, Input_D, seq_len
        x = self._conv0(x)        # ([32, 16, 25]) Batch, Input_D, seq_len
        x = x.permute(2,0,1)        # ([25, 32, 16]) seq_len, Batch, Input_D
        lstm_out, _ = self._lstm(x,self.hidden_state)       # ([25, 32, 40]) seq_len, Batch, Input_D
        lstm_out = lstm_out.permute(1, 2, 0)    # ([32, 40, 25])   # Batch, Input_D, seq_len
        flatten = self._flatten(lstm_out)
        score = self._classifier(flatten)
        return score, flatten

    def init_hidden(self):
        h0 = Variable(torch.zeros(1,self._batch_size,self._lstm_hid_dim))
        c0 = Variable(torch.zeros(1,self._batch_size,self._lstm_hid_dim))
        h0, c0 = h0.cuda(), c0.cuda()
        return (h0,c0)

    def _initailze(self, layer):
        if isinstance(layer, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_uniform_(layer.weight)