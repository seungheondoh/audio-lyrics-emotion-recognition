'''
feature_extraction.py

A file related with extracting feature.
For the baseline code it loads audio files and extract mel-spectrogram using Librosa.
Then it stores in the './feature' folder.
'''
import os
import numpy as np
import librosa
from pathlib import Path

import itertools
import pickle
from collections import Counter

import gluonnlp as nlp
import pandas as pd
import numpy as np

from src.data_manager import Vocab, split_sentence
from hparams import hparams

def load_lyrics(set_name):
    df = pd.read_csv('dataset/'+set_name+'.csv').loc[:, ['lyric']]
    lyrcis_list = []
    label_list = []

    for index in range(len(df)):
        lyrics = df.iloc[index]['lyric'] 
        lyrcis_list.append(lyrics)

    return lyrcis_list

def preprocessing(sentence):
    new_sentence = []
    for i in sentence.split():
        i.isalpha()
        new_sentence.append(i)
    return new_sentence

def melspectrogram(file_name, hparams):
	y, sr = librosa.load(file_name, hparams.sample_rate)
	S = librosa.stft(y, n_fft=hparams.fft_size, hop_length=hparams.hop_size, win_length=hparams.win_size)
	mel_basis = librosa.filters.mel(hparams.sample_rate, n_fft=hparams.fft_size, n_mels=hparams.num_mels)
	mel_S = np.dot(mel_basis, np.abs(S))
	mel_S = np.log10(1+10*mel_S)
	mel_S = mel_S.T
	return mel_S

def resize_array(array, length):
	resized_array = np.zeros((length, array.shape[1]))
	if array.shape[0] >= length:
		resized_array = array[:length]
	else:
		resized_array[:array.shape[0]] = array

	return resized_array

def save_audio_to_npy(dataset_path, feature_path):
	if not os.path.exists(feature_path):
		os.makedirs(feature_path)

	audio_path = os.path.join(dataset_path, 'MSD')
	audios = [path for path in os.listdir(audio_path)]
	
	for audio in audios:
		audio_abs = os.path.join(audio_path,audio)
		try:
			feature = melspectrogram(audio_abs, hparams)
			if len(feature) < 500:
				print("Feature length is less than 500")
		except:
			print("Cannot load audio {}".format(audio))
			continue
		feature = resize_array(feature, hparams.feature_length)
		fn = audio.split(".")[0]
		print(Path(feature_path)/(fn + '.npy'))
		np.save(Path(feature_path)/(fn + '.npy'), feature)


def build_vocab(hparams, types="fasttext", source="wiki.simple", min_freq=10):
	lyrics_train = load_lyrics('train')
	lyrics_valid = load_lyrics('valid')
	lyrics_test = load_lyrics('test')

	# Extract token in sentence
	total_vocab = lyrics_train + lyrics_valid + lyrics_test
	list_of_tokens = []
	for i in total_vocab:
		list_of_tokens.append(preprocessing(i))

	token_counter = Counter(itertools.chain.from_iterable(list_of_tokens))
	tmp_vocab = nlp.Vocab(
		counter=token_counter, min_freq=10, bos_token=None, eos_token=None)

	# connecting SISG embedding with vocab
	ptr_embedding = nlp.embedding.create(types, source=source)
	tmp_vocab.set_embedding(ptr_embedding)
	array = tmp_vocab.embedding.idx_to_vec.asnumpy()

	vocab = Vocab(
		tmp_vocab.idx_to_token,
		padding_token="<pad>",
		unknown_token="<unk>",
		bos_token=None,
		eos_token=None,
	)
	vocab.embedding = array

	# saving vocab
	with open(hparams.dataset_path+"/vocab.pkl", mode="wb") as io:
		pickle.dump(vocab, io)

    	
if __name__ == '__main__':
	# save_audio_to_npy(hparams.dataset_path, hparams.feature_path)
	build_vocab(hparams, types="fasttext", source ="wiki.simple", min_freq=5)