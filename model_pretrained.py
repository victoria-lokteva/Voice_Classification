import torch
import torchvision


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.inception = torchvision.models.inception_v3(pretrained=True)
        for parameter in self.inception.parameters():
            parameter.requires_grad = False
        in_features = self.inception.fc.in_features
        self.inception.fc = torch.nn.Linear(in_features=in_features, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.inception(x).logits
        x = self.sigmoid(x)
        return x
