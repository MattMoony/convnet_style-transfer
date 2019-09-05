import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import namedtuple

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg_fs = models.vgg19(pretrained=True).features

        self.sl1 = nn.Sequential()
        self.sl2 = nn.Sequential()
        self.sl3 = nn.Sequential()
        self.sl4 = nn.Sequential()
        self.sl5 = nn.Sequential()

        for i in range(4):
            self.sl1.add_module(str(i), vgg_fs[i])
        for i in range(4, 9):
            self.sl2.add_module(str(i), vgg_fs[i])
        for i in range(9, 18):
            self.sl3.add_module(str(i), vgg_fs[i])
        for i in range(18, 27):
            self.sl4.add_module(str(i), vgg_fs[i])
        for i in range(27, 36):
            self.sl5.add_module(str(i), vgg_fs[i])

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        h = self.sl1(x)
        h1 = h
        h = self.sl2(h)
        h2 = h
        h = self.sl3(h)
        h3 = h
        h = self.sl4(h)
        h4 = h
        h = self.sl5(h)
        h5 = h

        return_tuple = namedtuple('hidden_states', ['h1', 'h2', 'h3', 'h4', 'h5'])
        ret = return_tuple(h1, h2, h3, h4, h5)

        return ret

def show_img(y):
    if len(y.size()) >= 4:
        y = y.view(*y.shape[-2:], y.shape[-3])
    elif len(y.size()) == 3:
        y = y.view(*y.shape[1:], y.shape[0])
    else:
        return

    plt.imshow(y.detach().cpu())
    plt.show()

def norm(b):
    mean    = torch.Tensor([0.485, 0.456, 0.406]).cuda().view(3, 1, 1)
    std     = torch.Tensor([0.229, 0.224, 0.225]).cuda().view(3, 1, 1)
    return (b - mean) / std

def train(model, content_img, lr=1, reg_w=1e-5):
    content = torch.from_numpy(np.array(Image.open(content_img))).cuda().float()
    content = content / 255.
    content = content.view(1, content.size()[2], *content.size()[:2])

    content_fts = model(norm(content))

    img = torch.rand(*content.size(), requires_grad=True, device='cuda')
    optimizer = optim.Adam([img], lr=lr)
    mse_loss = nn.MSELoss()

    plt.ion()

    try:
        i = 0

        while True:
            optimizer.zero_grad()
            actvs = model(norm(img))

            lss = mse_loss(actvs.h2, content_fts.h2) +\
                    reg_w * (
                        torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])) +
                        torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
                    )

            lss.backward()
            optimizer.step()

            if i % 5 == 0:
                plt.title('Iter#{:04d}'.format(i))
                plt.imshow(img.detach().cpu().view(*img.shape[2:], img.shape[1]))
                plt.pause(1e-3)

            print('[iter#{:04d}]: Loss\t-> {:}'.format(i, lss.item()))
            i += 1
    except KeyboardInterrupt:
        pass

    plt.ioff()
    
    plt.subplot(121)
    plt.title('Original')
    plt.imshow(content.cpu().view(*content.size()[2:], content.size()[1]))

    plt.subplot(122)
    plt.title('Reproduced')
    plt.imshow(img.detach().cpu().view(*img.size()[2:], img.size()[1]))

    plt.show()

def main():
    model = VGG()
    model.cuda()

    train(model, 'media/content/woman-smiling-scaled.jpg', lr=0.3)

if __name__ == '__main__':
    main()