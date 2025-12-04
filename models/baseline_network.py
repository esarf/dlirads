import torch.nn as nn
import torch.nn.functional as F
import torch


class SmallNet(nn.Module):

    def __init__(self,
                 pretrained: bool = False,
                 segmask: bool = False,
                 num_classes: int = 2,
                 mode: str = 'encoder',
                 rep_dim: int = 512,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 input_dim: int = 2):
        super().__init__()
        self.mode = mode
        if pretrained:
            self.conv1 = nn.Conv2d(3, 32, 5)
        elif segmask:
            self.conv1 = nn.Conv2d(input_dim, 32//2, 5)
        else:
            self.conv1 = nn.Conv2d(1, 32//2, 5)

        self.features_conv = nn.Sequential(self.conv1,
                                           nn.ReLU(),
                                           nn.MaxPool2d(2, 2),
                                           # nn.Dropout(p=0.1),
                                           nn.Conv2d(32//2, 64//2, 5),
                                           nn.ReLU(),
                                           nn.MaxPool2d(2, 2),
                                           # nn.Dropout(p=0.1),
                                           nn.Conv2d(64//2, 128//2, 5),
                                           nn.ReLU(),
                                           nn.MaxPool2d(2, 2),
                                           # nn.Dropout(p=0.1),
                                           nn.Conv2d(128//2, 256//2, 5),
                                           nn.ReLU(),
                                           nn.MaxPool2d(2, 2),
                                           # nn.Dropout(p=0.1),
                                           nn.Conv2d(256//2, rep_dim, 5),
                                           nn.ReLU()
                                           )

        self.pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(rep_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.fc_cl = nn.Linear(hidden_dim, output_dim)

        #self.classifier = nn.Linear(rep_dim, num_classes)

    def forward(self, x, mode=None):
        if mode is None:
            mode = self.mode
        x = self.features_conv(x)
        x = self.pool(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        if mode == 'representation':
            return x.squeeze(dim=1)
        elif mode == 'encoder':
            x = F.relu(self.fc1(x))
            x = self.fc_cl(x)
        elif mode == 'classifier':
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        return x.squeeze(dim=1)


class SmallNet3D(nn.Module):

    def __init__(self,
                 pretrained: bool = False,
                 num_classes: int = 2,
                 mode: str = 'classifier',
                 rep_dim: int = 512,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 in_channels:int = 1):
        super().__init__()
        self.mode = mode
        self.conv1 = nn.Conv3d(in_channels, 32 //4, 5, padding='same')

        self.features_conv = nn.Sequential(self.conv1,
                                           nn.ReLU(),
                                           nn.MaxPool3d((1, 2, 2)),
                                           #nn.Dropout(p=0.1),
                                           nn.Conv3d(32//4, 64//4, 5, padding='same'),
                                           nn.ReLU(),
                                           nn.MaxPool3d((1, 2, 2)),
                                           #nn.Dropout(p=0.1),
                                           nn.Conv3d(64//4, 128//4, 5, padding='same'),
                                           nn.ReLU(),
                                           nn.MaxPool3d((1, 2, 2)),
                                           #nn.Dropout(p=0.1),
                                           nn.Conv3d(128 //4, 256 //4, 5, padding='same'),
                                           nn.ReLU(),
                                           nn.MaxPool3d((1, 2, 2)),
                                           #nn.Dropout(p=0.1),
                                           nn.Conv3d(256//4, rep_dim, 5, padding='same'),
                                           nn.ReLU()
                                           )

        self.pool = nn.MaxPool3d((1, 2, 2))
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(rep_dim#+1
                             , hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.fc_cl = nn.Linear(hidden_dim, output_dim)
        self.fc_hcc = nn.Linear(num_classes#+1
                                ,
                                1)
        self.dropout = nn.Dropout(p=0.1)

        # self.classifier = nn.Linear(rep_dim, num_classes)

    def forward(self, x, s=None, mode=None):
        if mode is None:
            mode = self.mode
        x = self.features_conv(x)
        x = self.pool(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        if mode == 'representation':
            #x = torch.cat((x, torch.unsqueeze(s, dim=1)), dim=1).to(torch.float)
            return x.squeeze(dim=1)
        elif mode == 'encoder':
            x = F.relu(self.fc1(x))
            x = self.fc_cl(x)
        elif mode == 'classifier':
            #x = torch.cat((x, torch.unsqueeze(s, dim=1)), dim=1).to(torch.float)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            #xsize = torch.cat((x, torch.unsqueeze(s, dim=1)), dim=1).to(torch.float)
            #y_hcc = self.fc_hcc(x)
            y_hcc = x[:, -1]
            return x.squeeze(dim=1), y_hcc#.squeeze(dim=1)
        return x.squeeze(dim=1)




class BaselineNet3D(nn.Module):

    def __init__(self,
                 pretrained: bool = False,
                 num_classes: int = 2,
                 mode: str = 'encoder',
                 rep_dim: int = 512,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 in_channels:int = 1):
        super().__init__()
        self.mode = mode
        self.conv1 = nn.Conv3d(in_channels, 32, 5, padding='same')

        self.features_conv = nn.Sequential(self.conv1,
                                           nn.ReLU(),
                                           nn.MaxPool3d((1, 2, 2)),
                                           #nn.Dropout(p=0.1),
                                           nn.Conv3d(32, 64, 5, padding='same'),
                                           nn.ReLU(),
                                           nn.MaxPool3d((1, 2, 2)),
                                           #nn.Dropout(p=0.1),
                                           nn.Conv3d(64, 128, 5, padding='same'),
                                           nn.ReLU(),
                                           nn.MaxPool3d((1, 2, 2)),
                                           #nn.Dropout(p=0.1),
                                           nn.Conv3d(128, 256, 5, padding='same'),
                                           nn.ReLU(),
                                           nn.MaxPool3d((1, 2, 2)),
                                           #nn.Dropout(p=0.1),
                                           nn.Conv3d(256, rep_dim, 5, padding='same'),
                                           nn.ReLU()
                                           )

        self.pool = nn.MaxPool3d((1, 2, 2))
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(rep_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.fc_cl = nn.Linear(hidden_dim, output_dim)

        # self.classifier = nn.Linear(rep_dim, num_classes)

    def forward(self, x, mode=None):
        if mode is None:
            mode = self.mode
        x = self.features_conv(x)
        x = self.pool(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        if mode == 'representation':
            return x.squeeze(dim=1)
        elif mode == 'encoder':
            x = F.relu(self.fc1(x))
            x = self.fc_cl(x)
        elif mode == 'classifier':
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        return x.squeeze(dim=1)





class BaselineNet(nn.Module):

    def __init__(self,
                 pretrained: bool = False,
                 segmask: bool = False,
                 num_classes: int = 2,
                 mode: str = 'encoder',
                 rep_dim: int = 512,
                 hidden_dim: int = 256,
                 output_dim: int = 128):
        super().__init__()
        self.mode = mode
        if pretrained:
            self.conv1 = nn.Conv2d(3, 32, 5)
        elif segmask:
            self.conv1 = nn.Conv2d(2, 32, 5)
        else:
            self.conv1 = nn.Conv2d(1, 32, 5)

        self.features_conv = nn.Sequential(self.conv1,
                                           nn.ReLU(),
                                           nn.MaxPool2d(2, 2),
                                           # nn.Dropout(p=0.1),
                                           nn.Conv2d(32, 64, 5),
                                           nn.ReLU(),
                                           nn.MaxPool2d(2, 2),
                                           # nn.Dropout(p=0.1),
                                           nn.Conv2d(64, 128, 5),
                                           nn.ReLU(),
                                           nn.MaxPool2d(2, 2),
                                           # nn.Dropout(p=0.1),
                                           nn.Conv2d(128, 256, 5),
                                           nn.ReLU(),
                                           nn.MaxPool2d(2, 2),
                                           # nn.Dropout(p=0.1),
                                           nn.Conv2d(256, rep_dim, 5),
                                           nn.ReLU()
                                           )

        self.pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(rep_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.fc_cl = nn.Linear(hidden_dim, output_dim)

        #self.classifier = nn.Linear(rep_dim, num_classes)

    def forward(self, x, mode=None):
        if mode is None:
            mode = self.mode
        x = self.features_conv(x)
        x = self.pool(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        if mode == 'representation':
            return x.squeeze(dim=1)
        elif mode == 'encoder':
            x = F.relu(self.fc1(x))
            x = self.fc_cl(x)
        elif mode == 'classifier':
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        #elif mode == 'frozen_classifier':
        #    x = self.classifier(x)
        return x.squeeze(dim=1)



