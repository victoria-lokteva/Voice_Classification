import torch
import torchaudio
import pandas as pd
import torchvision.transforms as transforms


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, training=True, sample_rate=24000, n_mels=64):
        dataset = torchaudio.datasets.LIBRITTS('.', download=True)
        # Разобъём на тренировочную и валидационную выборки
        train_size = 26589
        validation_size = len(dataset) - train_size
        train, validation = torch.utils.data.random_split(dataset, [train_size, validation_size])
        if training:
            self.data = train
        else:
            self.data = validation

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.resampling = torchaudio.transforms.Resample(new_freq=sample_rate)
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((299, 299)), transforms.ToTensor()])

        df = pd.read_csv('speakers.tsv', sep='\t', index_col=None).reset_index()
        self.speakers = dict(zip(df['index'], df['READER']))

    def mel_spectr(self, wave):
        # Преобразование Фурье с логарифмической шкалой частот
        transform = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels, sample_rate=self.sample_rate)
        # Переведем амплитуду в децибелы
        to_db = torchaudio.transforms.AmplitudeToDB()

        spec = transform(wave)
        spec = to_db(spec)
        spec = torch.squeeze(spec)
        return spec

    def spec_to_img(self, spec):
        mean, std = torch.mean(spec), torch.std(spec)
        # отцентрируем и нормализуем спектр в диапазоне от 0 до 255
        spec = (spec - mean) / (std + 0.000001)
        min_val, max_val = torch.min(spec), torch.max(spec)
        spec = 255 * (spec - min_val) / (max_val - min_val)
        return spec

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        item = list(self.data[index])

        waveform = item[0]

        waveform = self.resampling(waveform)
        waveform = self.mel_spectr(waveform)
        waveform = self.spec_to_img(waveform)
        waveform = self.transform(waveform)
        speaker_id = item[4]
        gender = self.speakers[speaker_id]
        if gender == 'M':
            gender = 1.
        elif gender == 'F':
            gender = 0.
        return waveform, gender
