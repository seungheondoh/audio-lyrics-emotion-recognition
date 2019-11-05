import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import numpy as np

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.LSTM):
    #             for p in m.named_parameters():
    #                 if 'weight' in p[0]:
    #                     nn.init.xavier_normal_(p[1]).cuda()

class MultiNet(MyModule):
    def __init__(self, hparams ,lstm_h=50, lstm_l=1, n_out=1, modal='audio'):
        super().__init__()
        self.modal = modal

        if modal=='audio':
            self.audionet = AudioNet(hparams)
        elif modal=='lyrics':
            self.lyricsnet = LyricsNet(hparams)
        elif modal=='multi':
            self.audionet = AudioNet(hparams)
            self.lyricsnet = LyricsNet(hparams)
        else:
            raise ValueError('Modal shoud be specified.')

        self._classifier = nn.Sequential(nn.Linear(in_features=976, out_features=64),
                                        nn.Tanh(),
                                        nn.Dropout(),
                                        nn.Linear(in_features=64, out_features=1),
                                        nn.Tanh())        
        # self.init_weights()

    def forward(self, audio, lyrics):
        if self.modal=='audio':
            f = self.audionet(audio)
        elif self.modal=='lyrics':
            f = self.lyricsnet(lyrics)
        elif self.modal=='multi':
            f_v = self.lyricsnet(lyrics)
            f_a = self.audionet(audio)
            print(f_v.shape, f_a.shape)
            f = torch.cat((f_a, f_v), 2)
            print(f.shape)
        
        # output = self._classifier(f)
        return f
            
class AudioNet(MyModule):
    def __init__(self, hparams):
        super(AudioNet, self).__init__()
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
                                        nn.Dropout(),
                                        nn.Linear(in_features=64, out_features=1),
                                        )

        self.apply(self._init_weights)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv0(x)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self._classifier(x)
        return x
        
    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)

        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

class LyricsNet(MyModule):
    def __init__(self,hparams,n_classes=1):
        super(LyricsNet,self).__init__()
        self.embeddings = self._load_embeddings(hparams.vocab_size,hparams.emb_dim)
        self.conv0 = nn.Sequential(
                        nn.Conv1d(hparams.max_len, 16, kernel_size=2, stride=1, padding=1),
                        nn.MaxPool1d(2, stride=2))
        self.lstm = torch.nn.LSTM(hparams.emb_dim, hparams.lstm_hid_dim, batch_first=False)
        self._classifier = nn.Sequential(nn.Linear(in_features=hparams.lstm_hid_dim, out_features=64),
                                        nn.Tanh(),
                                        nn.Dropout(),
                                        nn.Linear(in_features=64, out_features=1)
                                        )
        self.batch_size = hparams.batch_size
        self.max_len = hparams.max_len
        self.lstm_hid_dim = hparams.lstm_hid_dim
        self.hidden_state = self.init_hidden()

    def _load_embeddings(self,vocab_size=20000, emb_dim=100):
        word_embeddings = torch.nn.Embedding(vocab_size,emb_dim,padding_idx=0)
        return word_embeddings
       
    def init_hidden(self):
        h0 = Variable(torch.zeros(1,self.max_len,self.lstm_hid_dim))
        c0 = Variable(torch.zeros(1,self.max_len,self.lstm_hid_dim))
        h0, c0 = h0.cuda(), c0.cuda()
        return (h0,c0)
       
    def forward(self,x):
        embeddings = self.embeddings(x)
        x = self.conv0(embeddings)
        outputs, self.hidden_state = self.lstm(x.view(x.size(0),-1),self.hidden_state)
        # outputs = outputs.view(outputs.size(0), -1)
        print(outputs.shape)
        # x = self._classifier(outputs)
        return outputs