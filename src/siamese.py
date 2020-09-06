'''Siamese Network Architecture'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# torch cuda tensor configuration
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class SiameseNetwork(nn.Module):
    r'''Siamese Network Class'''

    def __init__(self):
        r'''Constructor of the SiameseNetwork Class'''

        super(SiameseNetwork, self).__init__()

        self.ConvLayer = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8))

        self.FlattenLayer = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 5))

    def forward_once(self, x):
        x = self.ConvLayer(x)
        x = x.view(x.size(0), -1).contiguous()
        x = self.FlattenLayer(x)
        return x

    def forward(self, inp1, inp2):
        out1 = self.forward_once(inp1)
        out2 = self.forward_once(inp2)
        return out1, out2


class ContrastiveLoss(torch.nn.Module):
    '''Contrastive Loss Function for Siamese Network'''

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        euclidean_dist = F.pairwise_distance(out1, out2, keepdim=True)
        loss = torch.mean((1-label)*torch.pow(euclidean_dist, 2) +
                          label*torch.pow(torch.clamp((self.margin -
                                                       euclidean_dist),
                                                      min=0.0), 2))
        return loss
