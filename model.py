import numpy as np
import torch
import torch.optim as optim
import torchvision
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.inception = model = torchvision.models.inception_v3(pretrained=False)
        in_features = self.inception.fc.in_features
        self.inception.fc = torch.nn.Linear(in_features=in_features, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.inception(x).logits
        x = self.sigmoid(x)
        return x

    def random_seed(rs=10):
        np.random.seed(rs)
        torch.manual_seed(rs)
        torch.cuda.manual_seed(rs)
        torch.backends.cudnn.deterministic = True


def random_seed(rs=10):
    np.random.seed(rs)
    torch.manual_seed(rs)
    torch.cuda.manual_seed(rs)
    torch.backends.cudnn.deterministic = True


def train(dataloader, device_name, model, lr=0.001, num_epoch=30):
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(num_epoch)):
        train_loader = iter(dataloader)
        train_loss = 0
        for batch, (X, Y) in enumerate(tqdm(train_loader)):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_func(pred, Y.unsqueeze(1))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        if epoch % 15 == 0:
            print("Epoch:  ", epoch, "    Train Loss:  ", train_loss)

    return model


def test(model, dataloader, device, threshold=0.5):
    predicted_labels = []
    predicted_probabilities = []
    labels = []
    test_loss = 0
    model = model.to(device)
    test_loader = iter(dataloader)
    model = model.eval()
    with torch.no_grad:
        for batch, (X, Y) in enumerate(tqdm(test_loader)):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            pred_labels = pred >= threshold
            loss = torch.nn.CrossEntropyLoss(pred, Y.unsqueeze(1))

            labels.extend(Y.to_list)
            predicted_labels.extend(pred_labels.squeeze().to_list)
            predicted_probabilities.extend(pred.squeeze().to_list)
            test_loss += loss.item()

    accuracy = accuracy_score(labels, predicted_labels)
    rocauc = roc_auc_score(labels, predicted_probabilities)
    print('Test Loss: %0.3f %% ' % (test_loss),
          'Accuracy: %0.3f %% ' % (accuracy),
          'RocAUC: %0.3f %% ' % (rocauc))
