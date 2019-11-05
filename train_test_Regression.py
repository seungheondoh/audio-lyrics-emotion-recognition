'''
train_test.py

A file for training model for genre classification.
Please check the device in hparams.py before you run this code.
'''
# torch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
# other
import pickle
import numpy as np
from sklearn.metrics import r2_score
# private
import data_loader
from hparams import hparams
import utils

from src.data_manager import Vocab, split_sentence, PadSequence, Tokenizer
from models.net import MultiNet


# Wrapper class to run PyTorch model
class Runner(object):
    def __init__(self, hparams, vocab:Vocab):
        self.model = MultiNet(hparams, vocab, modal=hparams.modal)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=hparams.learning_rate, momentum=hparams.momentum)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=hparams.factor, patience=hparams.patience, verbose=True)
        self.learning_rate = hparams.learning_rate
        self.stopping_rate = hparams.stopping_rate
        self.device = hparams.device
        self.criterion = torch.nn.MSELoss()

    # Running model for train, test and validation. mode: 'train' for training, 'eval' for validation and test
    def run(self, dataloader, mode='train'):
        self.model.train() if mode is 'train' else self.model.eval()
        epoch_loss = 0

        all_prediction = []
        all_label = []
        model, floatTensor, longTensor = utils.set_device(self.model, self.device)

        for batch, (audio, lyrics, labels) in enumerate(dataloader):
            
            audio = audio.type(floatTensor)
            lyrics = lyrics.type(longTensor)
            labels = labels.type(floatTensor)
                        
            prediction = self.model(audio, lyrics)
            prediction = torch.squeeze(prediction,1)
            loss = self.criterion(prediction, labels)

            all_prediction.extend(prediction.cpu().detach().numpy())
            all_label.extend(labels.cpu().detach().numpy())

            if mode is 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss += prediction.size(0)*loss.item()

        epoch_loss = epoch_loss/len(dataloader.dataset)

        return all_prediction, all_label, epoch_loss

    # Early stopping function for given validation loss
    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        stop = self.learning_rate < self.stopping_rate
        return stop


def main():

    # tokenizer
    with open(hparams.dataset_path+"/vocab.pkl", mode='rb') as io:
        vocab = pickle.load(io)
    pad_sequence = PadSequence(length=hparams.max_len, pad_val=vocab.to_indices(vocab.padding_token))
    tokenizer = Tokenizer(vocab=vocab, split_fn=split_sentence, pad_fn=pad_sequence)

    # data loader
    train_loader, valid_loader, test_loader = data_loader.get_dataloader(hparams)
    runner = Runner(hparams, vocab=tokenizer.vocab)

    print('Training on ' + str(hparams.device))
    for epoch in range(hparams.num_epochs):
        train_pred, train_label, train_loss = runner.run(train_loader, 'train')
        valid_pred, valid_label, valid_loss = runner.run(valid_loader, 'eval')

        train_score = r2_score(train_label, train_pred)
        valid_score = r2_score(valid_label, valid_pred)

        print("[Epoch %d/%d] [Train Loss: %.4f] [Train R2: %.4f] [Valid Loss: %.4f] [Valid R2: %.4f]" %
              (epoch + 1, hparams.num_epochs, train_loss, train_score, valid_loss, valid_score))

        if runner.early_stop(valid_loss, epoch + 1):
            break

    test_pred, test_label, test_loss = runner.run(test_loader, 'eval')
    print("Training Finished")
    test_score = r2_score(test_label, test_pred)
    print("Test R2:  %.4f" %(test_score))

if __name__ == '__main__':
    main()
