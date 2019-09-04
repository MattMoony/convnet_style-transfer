from skimage import io
import matplotlib.pyplot as plt
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
import torchvision.models as models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        vgg_feats = models.vgg16(pretrained=True).features

        self.sl_1 = torch.nn.Sequential()
        self.sl_2 = torch.nn.Sequential()
        self.sl_3 = torch.nn.Sequential()
        self.sl_4 = torch.nn.Sequential()
        self.sl_5 = torch.nn.Sequential()

        for i in range(4):
            self.sl_1.add_module(str(i), vgg_feats[i])
        for i in range(4, 9):
            self.sl_2.add_module(str(i), vgg_feats[i])
        for i in range(9, 16):
            self.sl_3.add_module(str(i), vgg_feats[i])
        for i in range(16, 23):
            self.sl_4.add_module(str(i), vgg_feats[i])
        for i in range(23, 30):
            self.sl_5.add_module(str(i), vgg_feats[i])

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        h = self.sl_1(x)
        h_state_1 = h
        h = self.sl_2(h)
        h_state_2 = h
        h = self.sl_3(h)
        h_state_3 = h
        h = self.sl_4(h)
        h_state_4 = h
        h = self.sl_5(h)
        h_state_5 = h

        return [h_state_1, h_state_2, h_state_3, h_state_4, h_state_5]

def show_t_img(t):
    plt.imshow(t.detach().cpu().view(*t.shape[2:], t.shape[1]))
    plt.show()

def gram_mat(f):
    (b, c, h, w) = f.shape
    f = f.view(b, c, h * w)
    g = f.bmm(f.transpose(1, 2)) / (c * h * w)
    return g

def train(model, cont_img, style_img, transform, args):
    img = torch.randn(1, *cont_img.size(), requires_grad=True, device='cuda')
    # img = torch.ones(1, *cont_img.size(), requires_grad=True, device='cuda')

    optimizer = optim.Adam([img], lr=args.lr)
    mse_loss = nn.MSELoss()

    cont_feat = model(cont_img.view(1, *cont_img.size()))
    for f in cont_feat:
        f.requires_grad_(False)
        
    style_feat = model(style_img.view(1, *style_img.size()))
    gram_mats = [gram_mat(f) for f in style_feat]
    for g in gram_mats:
        g.requires_grad_(False)

    try:
        for i in range(args.iters):
            optimizer.zero_grad()
            out = model(img)
            # out = model(transform(img.view(*img.shape[1:])).view(*img.shape))

            content_loss = args.cont_w * mse_loss(out[1], cont_feat[1])

            style_loss = mse_loss(gram_mat(out[0]), gram_mats[0])
            for j in range(1, len(out)):
                style_loss += mse_loss(gram_mat(out[j]), gram_mats[j])
            style_loss *= args.style_w

            reg_loss = args.reg_w * (
                torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])) +
                torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
            )

            total_loss = content_loss + style_loss + reg_loss
            total_loss.backward()
            optimizer.step()

            print('[epoch#{:04d}]: Total-Loss\t-> {:} ... '.format(i, total_loss.item()))
            print('              Content-Loss\t-> {:} ... '.format(content_loss.item()))
            print('              Style-Loss\t-> {:} ... '.format(style_loss.item()))
            print('              Reg-Loss\t\t-> {:} ... '.format(reg_loss.item()))
            print('='*64)
    except KeyboardInterrupt:
        pass

    return img

def main():
    model = Net()
    model.to('cuda')

    style_img = io.imread('media/style/gustav-klimt-kiss.jpg')
    style_img = style_img.reshape(3, *style_img.shape[:-1])
    cont_img = io.imread('media/content/parrots-kiss-scaled.jpg')
    cont_img = cont_img.reshape(3, *cont_img.shape[:-1])

    style_img = torch.from_numpy(style_img).to(device='cuda', dtype=torch.float)
    cont_img = torch.from_numpy(cont_img).to(device='cuda', dtype=torch.float)

    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])

    style_img = transform(style_img / 255.0)
    cont_img = transform(cont_img / 255.0)

    arguments = namedtuple('args', ['lr', 'iters', 'cont_w', 'style_w', 'reg_w'])
    args = arguments(0.06, 2048, 1., 1., 0.)

    img = train(model, cont_img, style_img, transform, args)
    show_t_img(img)

    print(img.min().item())
    print(img.mean().item())
    print(img.max().item())
    
if __name__ == '__main__':
    main()