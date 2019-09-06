import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models
import torchvision.transforms as transforms

import sys
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

def comp_gram(f):
    (b, c, h, w) = f.shape
    f = f.view(b, c, h * w)
    g = f.bmm(f.transpose(1, 2)) 
    g = g / (c * h * w)
    return g

def norm(b):
    mean    = torch.Tensor([0.485, 0.456, 0.406]).cuda().view(3, 1, 1)
    std     = torch.Tensor([0.229, 0.224, 0.225]).cuda().view(3, 1, 1)
    return (b - mean) / std

def load_img(p):
    img = Image.open(p).convert('RGB')
    im_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img = im_transform(img).unsqueeze(0).cuda()

    return img

def to_img(t):
    img = t.cpu().clone().detach()
    img = img.numpy().squeeze()
    img = img.transpose(1, 2, 0)
    img = img.clip(0, 1)

    return img

def train_s(model, style, w=224, h=224, lr=1, iters=64):
    style_fts = model(norm(style))
    style_gms = [comp_gram(f) for f in style_fts]

    img = torch.rand(*style.size()[:2], h, w, requires_grad=True, device='cuda')
    optimizer = optim.Adam([img], lr=lr)
    mse_loss = nn.MSELoss()

    lss = 0.

    for i in range(iters):
        optimizer.zero_grad()
        actvs = model(norm(img))

        lss = 0.
        for f, gm in zip(actvs, style_gms):
            g = comp_gram(f)
            lss += mse_loss(g, gm)

        lss.backward()
        optimizer.step()

    return lss.item()

def find_lr(model, style_img, iters=64, n=32, w=224, h=224, top=1, bot=1e-6):
    style = torch.from_numpy(np.array(Image.open(style_img))).cuda().float()
    style = style / 255.
    style = style.view(1, style.size()[2], *style.size()[:2])

    lrs = torch.rand(n) * (top - bot) + bot

    min_lss = sys.maxsize
    min_lr = 0.

    for lr in lrs:
        lss = train_s(model, style, w, h, lr, iters)

        if lss < min_lss:
            min_lss = lss
            min_lr = lr

            print('[+] New min-lr: {:} ... '.format(lr))

    return min_lr

def train(model, style_img, w=224, h=224, lr=1, reg_w=1e-8):
    style = load_img(style_img)
    
    style_fts = model(norm(style))
    style_gms = [comp_gram(f) for f in style_fts]

    img = torch.rand(*style.size()[:2], h, w, requires_grad=True, device='cuda')
    optimizer = optim.Adam([img], lr=lr)
    mse_loss = nn.MSELoss()

    plt.ion()
    
    try:
        i = 0
        while True:
            optimizer.zero_grad()
            actvs = model(norm(img))

            lss = 0.
            for f, gm in zip(actvs, style_gms):
                g = comp_gram(f)
                lss += mse_loss(g, gm)
                
            lss += reg_w * (
                        torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])) +
                        torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
                    )

            lss.backward()
            optimizer.step()

            if i % 5 == 0:
                plt.title('Iter#{:04d}'.format(i))
                plt.imshow(to_img(img))
                plt.pause(1e-3)

            print('[iter#{:04d}]: Loss\t-> {:}'.format(i, lss.item()))
            i += 1
    except KeyboardInterrupt:
        pass

    plt.ioff()
    
    plt.subplot(121)
    plt.title('Original')
    plt.imshow(to_img(style))

    plt.subplot(122)
    plt.title('Style')
    plt.imshow(to_img(img))

    plt.show()

def main():
    model = VGG()
    model.cuda()

    style_img = 'media/style/gustav-klimt-kiss.jpg'
    w = 400
    h = 224

    # lr = find_lr(model, style_img, w=w, h=h)
    lr = .3
    train(model, style_img, w=w, h=h, lr=lr)

if __name__ == '__main__':
    main()