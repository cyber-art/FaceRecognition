'''Trainer Class for the Siamese Network on Face Recognition'''

import torch
import torch.nn as nn
from torch import optim
import torchvision.datasets as dset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
try:
    from dataset import ORLFaceDataset
    from siamese import SiameseNetwork, ContrastiveLoss
except ImportError:
    from src.dataset import ORLFaceDataset
    from src.siamese import SiameseNetwork, ContrastiveLoss


class SiameseTrainer():
    '''Face Recognition Network Trainer Class'''

    def __init__(self, network, train_path, test_path):
        '''Constructor of the SiameseTrainer Class'''

        self.SiameseNetwork = network
        self.criterion = ContrastiveLoss()
        self.CUDA = torch.cuda.is_available()
        self.optimizer = optim.Adam(self.SiameseNetwork.parameters(),
                                    lr=0.0005)
        if self.CUDA:
            self.SiameseNetwork = self.SiameseNetwork.cuda()
        if torch.cuda.device_count() > 1:
            self.SiameseNetwork = nn.DataParallel(self.SiameseNetwork)
        train_dset = dset.ImageFolder(root=train_path)
        test_dset = dset.ImageFolder(root=test_path)
        transform = transforms.Compose([transforms.Resize((100, 100)),
                                        transforms.ToTensor()])
        self.train_dset = ORLFaceDataset(train_dset,
                                         transform=transform,
                                         should_invert=False)
        self.test_dset = ORLFaceDataset(test_dset,
                                        transform=transform,
                                        should_invert=False)

    @staticmethod
    def progress_bar(curr_epoch, curr_batch, batch_num, loss):
        percent = curr_batch/batch_num
        last = int((percent*1000) % 10)
        percent = round(percent*100)
        bar = 'Epoch: {:3d} '.format(curr_epoch)
        bar += 'Batch: {:3d} '.format(curr_batch)
        bar += 'Loss: {:.4f} '.format(loss)
        bar += '|' + '#' * int(percent)
        if curr_batch != batch_num:
            bar += '{}'.format(last)
            bar += ' ' * (100-int(percent)) + '|'
            print('\r'+bar, end='')
        else:
            bar += '#'
            bar += ' ' * (100-int(percent)) + '|'
            print('\r'+bar)

    def save_state_dict(self, dict_path):
        torch.save(self.SiameseNetwork.state_dict(), dict_path)

    def load_state_dict(self, dict_path):
        self.SiameseNetwork.load_state_dict(torch.load(dict_path))

    def train(self, batch_size, epochs):
        train_dloader = DataLoader(self.train_dset,
                                   shuffle=True,
                                   num_workers=8,
                                   batch_size=batch_size)

        counter = []
        loss_history = []
        iteration_number = 0
        batch_num = self.train_dset.__len__()//batch_size

        for epoch in range(0, epochs):
            for i, data in enumerate(train_dloader, 0):
                img0, img1, label = data
                if self.CUDA:
                    img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                self.optimizer.zero_grad()
                output1, output2 = self.SiameseNetwork(img0, img1)
                loss = self.criterion(output1, output2, label)
                loss.backward()
                self.optimizer.step()
                self.progress_bar(epoch+1, i, batch_num, loss.item())
            iteration_number += 1
            counter.append(iteration_number)
            loss_history.append(loss.item())
        plt.plot(counter, loss_history)
        plt.savefig('src/statedicts/loss_graph.png')


if __name__ == '__main__':
    net = SiameseNetwork()
    train_path = '/home/adm1n/Datasets/ORLFace/faces/training/'
    test_path = '/home/adm1n/Datasets/ORLFace/faces/testing/'
    trainer = SiameseTrainer(net, train_path, test_path)
    trainer.train(batch_size=16, epochs=100)
    trainer.save_state_dict('statedicts/experiment1')
