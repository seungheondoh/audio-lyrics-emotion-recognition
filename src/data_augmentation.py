import librosa
import numpy as np
import os
from hparams import hparams

def get_genre(hparams):
    return hparams.clusters

def load_list(list_name, hparams):
    with open(os.path.join(hparams.dataset_path, list_name)) as f:
        file_names = f.read().splitlines()

    return file_names

def get_item(hparams, clusters):
    return librosa.util.find_files(hparams.dataset_path + '/' + str(clusters))


def readfile(file_name, hparams):
    y, sr = librosa.load(file_name, hparams.sample_rate)
    return y, sr


def change_pitch_and_speed(data):
    y_pitch_speed = data.copy()
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high=1)
    speed_fac = 1.0 / length_change
    tmp = np.interp(np.arange(0, len(y_pitch_speed), speed_fac), np.arange(0, len(y_pitch_speed)), y_pitch_speed)
    minlen = min(y_pitch_speed.shape[0], tmp.shape[0])
    y_pitch_speed *= 0
    y_pitch_speed[0:minlen] = tmp[0:minlen]
    return y_pitch_speed


def change_pitch(data, sr):
    y_pitch = data.copy()
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    y_pitch = librosa.effects.pitch_shift(y_pitch.astype('float64'), sr, n_steps=pitch_change,
                                          bins_per_octave=bins_per_octave)
    return y_pitch

def value_aug(data):
    y_aug = data.copy()
    dyn_change = np.random.uniform(low=1.5, high=3)
    y_aug = y_aug * dyn_change
    return y_aug


def add_noise(data):
    noise = np.random.randn(len(data))
    data_noise = data + 0.005 * noise
    return data_noise


def hpss(data):
    y_harmonic, y_percussive = librosa.effects.hpss(data.astype('float64'))
    return y_harmonic, y_percussive


def shift(data):
    return np.roll(data, 1600)


def stretch(data, rate=1):
    input_length = len(data)
    streching = librosa.effects.time_stretch(data, rate)
    if len(streching) > input_length:
        streching = streching[:input_length]
    else:
        streching = np.pad(streching, (0, max(0, input_length - len(streching))), "constant")
    return streching

def change_speed(data):
    y_speed = data.copy()
    speed_change = np.random.uniform(low=0.9, high=1.1)
    tmp = librosa.effects.time_stretch(y_speed.astype('float64'), speed_change)
    minlen = min(y_speed.shape[0], tmp.shape[0])
    y_speed *= 0
    y_speed[0:minlen] = tmp[0:minlen]
    return y_speed

def main():
    print('Augmentation')
    genres = get_genre(hparams)
    list_names = ['train_list.txt']
    # for list_name in list_names:
    #     file_names = load_list(list_name, hparams)
        # with open(os.path.join(hparams.dataset_path, list_name),'w') as f:
        #     for i in file_names:
        #         f.writelines(i+'\n')
        #         f.writelines(i.replace('.mp3', 'a.mp3' + '\n'))
        #         f.writelines(i.replace('.mp3', 'b.mp3' + '\n'))
        #         f.writelines(i.replace('.mp3', 'c.mp3' + '\n'))
        #         f.writelines(i.replace('.mp3', 'd.mp3' + '\n'))
        #         f.writelines(i.replace('.mp3', 'e.mp3' + '\n'))
        #         f.writelines(i.replace('.mp3', 'f.mp3' + '\n'))
        #         f.writelines(i.replace('.mp3', 'g.mp3' + '\n'))
        #         f.writelines(i.replace('.mp3', 'h.mp3' + '\n'))
        #         f.writelines(i.replace('.mp3', 'i.mp3' + '\n'))

    for genre in genres:
        item_list = get_item(hparams, genre)
        # print("check=",genre)
        # print("item_list=",item_list)
        for file_name in item_list:
            # print("filename",file_name)
        
            y, sr = readfile(file_name, hparams)
            data_noise = add_noise(y)
            data_roll = shift(y)
            data_stretch = stretch(y)
            pitch_speed = change_pitch_and_speed(y)
            pitch = change_pitch(y, hparams.sample_rate)
            speed = change_speed(y)
            value = value_aug(y)
            y_harmonic, y_percussive = hpss(y)
            y_shift = shift(y)

            save_path = os.path.join(file_name.split('MSD/')[0] + 'total_augmentation')
            save_name =file_name.split(genre + '/')[1]
            print(save_path ,save_name)

            librosa.output.write_wav(os.path.join(save_path, save_name), y,
                                     hparams.sample_rate)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.mp3', 'a.mp3')), data_noise,
                                     hparams.sample_rate)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.mp3', 'b.mp3')), data_roll,
                                     hparams.sample_rate)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.mp3', 'c.mp3')), data_stretch,
                                     hparams.sample_rate)
            # librosa.output.write_wav(os.path.join(save_path, save_name.replace('.mp3', 'd.mp3')), pitch_speed,
            #                          hparams.sample_rate)
            # librosa.output.write_wav(os.path.join(save_path, save_name.replace('.mp3', 'e.mp3')), pitch,
            #                          hparams.sample_rate)
            # librosa.output.write_wav(os.path.join(save_path, save_name.replace('.mp3', 'f.mp3')), speed,
            #                          hparams.sample_rate)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.mp3', 'g.mp3')), value,
                                     hparams.sample_rate)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.mp3', 'h.mp3')), y_percussive,
                                     hparams.sample_rate)
            # librosa.output.write_wav(os.path.join(save_path, save_name.replace('.mp3', 'i.mp3')), y_shift,
            #                          hparams.sample_rate)
        print('finished')


if __name__ == '__main__':
    main()

