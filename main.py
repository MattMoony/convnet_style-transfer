import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models
import torchvision.transforms as transforms

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import namedtuple
from argparse import ArgumentParser

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg_fs = models.vgg16(pretrained=True).features

        self.sl1 = nn.Sequential()
        self.sl2 = nn.Sequential()
        self.sl3 = nn.Sequential()
        self.sl4 = nn.Sequential()
        self.sl5 = nn.Sequential()

        for i in range(4):
            self.sl1.add_module(str(i), vgg_fs[i])
        for i in range(4, 9):
            self.sl2.add_module(str(i), vgg_fs[i])
        for i in range(9, 16):
            self.sl3.add_module(str(i), vgg_fs[i])
        for i in range(16, 23):
            self.sl4.add_module(str(i), vgg_fs[i])
        for i in range(23, 30):
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

def save_img(t, p):
    img = to_img(t)
    img = Image.fromarray((img * 255.).astype(np.uint8))
    img.save(p)

def train(model, content_img, style_img, lr=0.2, content_w=1, style_w=1, tv_w=1e-5, interval=5, verbose=True):
    content = load_img(content_img)
    style = load_img(style_img)

    content_fts = model(norm(content))

    style_fts = model(norm(style))
    style_gms = [comp_gram(f) for f in style_fts]

    img = torch.rand(*content.size(), requires_grad=True, device='cuda')
    optimizer = optim.Adam([img], lr=lr)
    mse_loss = nn.MSELoss()

    plt.ion()

    try:
        i = 0
        while True:
            torch.cuda.empty_cache()

            optimizer.zero_grad()
            actvs = model(norm(img))

            content_loss = content_w * mse_loss(actvs.h2, content_fts.h2)

            style_loss = 0.
            for f, gm in zip(actvs, style_gms):
                g = comp_gram(f)
                style_loss += mse_loss(g, gm)
            style_loss *= style_w

            tv_loss = tv_w * (
                        torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])) +
                        torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
                    )

            total_loss = content_loss + style_loss + tv_loss
            total_loss.backward()
            optimizer.step()

            if verbose and (i % interval == 0):
                plt.title('Iter#{:04d}'.format(i))
                plt.imshow(to_img(img))
                plt.pause(1e-3)

            print('[iter#{:04d}]: Loss\t-> {:}'.format(i, total_loss.item()))
            i += 1
    except KeyboardInterrupt:
        pass

    plt.ioff()

    if verbose:
        plt.subplot(131)
        plt.title('Content')
        plt.imshow(to_img(content))

        plt.subplot(133)
        plt.title('Style')
        plt.imshow(to_img(style))

        plt.subplot(132)
        plt.title('Result')
        plt.imshow(to_img(img))

        plt.show()

    return img

def check_paths(*args):
    true = []
    false = []

    for p in args:
        if os.path.isfile(p):
            true.append(p)
        else:
            false.append(p)

    return true, false

def main():
    parser = ArgumentParser()

    parser.add_argument('-c', '--content', dest='content', help='Location of the content image ... ', required=True)
    parser.add_argument('-s', '--style', dest='style', help='Location of the style image ... ', required=True)
    parser.add_argument('-d', '--destination', dest='destination', help='Destination of the result ... ')

    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', help='Display intermediate results? ')
    parser.add_argument('-i', '--interval', type=int, dest='interval', help='The interval to display current results ... ', 
                            default=10)

    parser.add_argument('--lr', dest='lr', type=float, help='Specify a learning rate ... ', default=0.1)
    parser.add_argument('--content-w', dest='content_w', type=float, help='Specify the content weight ... ', default=1)
    parser.add_argument('--style-w', dest='style_w', type=float, help='Specify the style weight ... ', default=10e5)
    parser.add_argument('--tv-w', dest='tv_w', type=float, help='Specify a total variation weight ... ', default=3e-4)

    args = parser.parse_args()

    y, n = check_paths(args.content, args.style)
    if len(n) > 0:
        print('The following files don\'t exist: ' + ', '.join(n))
        os._exit(1)

    if args.destination and (
        not os.path.isdir(os.path.dirname(args.destination)) or 
        os.path.isfile(args.destination) or
        not os.path.basename(args.destination).endswith('.jpg')
    ):
        print('The destination path is not valid ... (either non-existend directory, filename already present or non-jpg filename)')
        os._exit(1)

    model = VGG()
    model.cuda()

    result_img = train(model, args.content, args.style, 
                        lr=args.lr, content_w=args.content_w, style_w=args.style_w, tv_w=args.tv_w, 
                        verbose=args.verbose, interval=args.interval)

    if not args.destination:
        yN = input('Save result? [y/N] ')
        if yN in ['y', 'Y']:
            path = ''
            while (
                not os.path.isdir(os.path.dirname(path)) or 
                os.path.isfile(path) or
                not os.path.basename(path).endswith('.jpg')
            ):
                path = input('Enter the destination: ')

            save_img(result_img, path)
    else:
        save_img(result_img, args.destination)
    
if __name__ == '__main__':
    main()