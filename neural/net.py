from torch import nn


class Net(nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.scale_min = min
        self.scale_max = max

        self.fc1 = nn.Sequential(
            nn.Linear(300, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(0.20)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(0.20)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(500, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.20)
        )

        self.fc4 = nn.Sequential(
            nn.Linear(100, 9),
            nn.BatchNorm1d(9),
            nn.ReLU(),
            nn.Dropout(0.20)
        )

        self.fc5 = nn.Sequential(
            nn.Linear(9, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)

        x = x * (self.scale_max-self.scale_min) + self.scale_min
        x = x.squeeze(dim=1)

        return x