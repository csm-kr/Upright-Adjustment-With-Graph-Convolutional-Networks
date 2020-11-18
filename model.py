import torch
import torch.nn as nn
import torchvision.models as models
from layers import GraphConvolution
import torch.nn.functional as F


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(*list(models.densenet121(pretrained=True).features.children()))

        self.fc = torch.nn.Linear(1024*7*13, 2)
        self.fc.weight = nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x.view(-1, 1024 * 91))
        return x


class DenseNet_GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(*list(models.densenet121(pretrained=True).features.children()),
                                      nn.Sequential(
                                          nn.Conv2d(1024, 512, (3, 3), padding=(1, 1)),
                                          nn.BatchNorm2d(512),
                                          nn.ReLU(inplace=True),
                                      ),)

        self.gcn_1 = GraphConvolution(512, 256)
        self.gcn_2 = GraphConvolution(256, 128)
        self.gcn_3 = GraphConvolution(128, 64)
        self.gcn_4 = GraphConvolution(64, 4)
        self.gcn_5 = GraphConvolution(4, 1)

    def forward(self, x, adj):
        x = self.features(x)
        x = x.view(-1, 91, 512)  # [B, 91, 512]  # feature to graph feature
        x = F.relu(self.gcn_1(x, adj))
        x = F.relu(self.gcn_2(x, adj))
        x = F.relu(self.gcn_3(x, adj))
        x = F.relu(self.gcn_4(x, adj))
        x = self.gcn_5(x, adj)

        return x


if __name__ == "__main__":

    dense_net = DenseNet()
    img = torch.Tensor(2, 3, 221, 442).type(torch.float32)
    adj = torch.Tensor(2, 91, 91).type(torch.float32)
    print(dense_net(img).size())
    pytorch_total_params = sum(p.numel() for p in dense_net.features.parameters() if p.requires_grad)
    print('densenet101\' s param num : ', pytorch_total_params)

