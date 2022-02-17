import torch
from model import Model, train, test, random_seed
from preprocessing import AudioDataset

random_seed()

train_data = AudioDataset(training=True)
validation = AudioDataset(training=False)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(validation, batch_size=64)

model = Model()
model = train(train_loader, 'cuda:0', model, 0.001, 3)
model = test(test_loader, 'cuda:0', model)
