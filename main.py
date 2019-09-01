import torch
import torchvision.models as models
from skimage import io

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        vgg_feats = models.vgg16(pretrained=True).features

        self.sl_1 = torch.nn.Sequential()
        self.sl_2 = torch.nn.Sequential()
        self.sl_3 = torch.nn.Sequential()
        self.sl_4 = torch.nn.Sequential()

        for i in range(4):
            self.sl_1.add_module(str(i), vgg_feats[i])
        for i in range(4, 9):
            self.sl_2.add_module(str(i), vgg_feats[i])
        for i in range(9, 16):
            self.sl_3.add_module(str(i), vgg_feats[i])
        for i in range(16, 23):
            self.sl_4.add_module(str(i), vgg_feats[i])

    def forward(self, x):
        h_state_1 = self.sl_1(x)
        h_state_2 = self.sl_2(h_state_1)
        h_state_3 = self.sl_3(h_state_2)
        h_state_4 = self.sl_4(h_state_3)

        return dict(
            h_state_1 = h_state_1,
            h_state_2 = h_state_2,
            h_state_3 = h_state_3,
            h_state_4 = h_state_4
        )

def main():
    model = Net()
    model.to('cuda')

    style_img = io.imread('media/style/gustav-klimt-kiss-scaled.jpg').reshape(3, 256, 256)
    cont_img = io.imread('media/content/parrots-kiss-scaled.jpg').reshape(3, 256, 256)

    style_img = torch.from_numpy(style_img).to(device='cuda', dtype=torch.float)
    cont_img = torch.from_numpy(cont_img).to(device='cuda', dtype=torch.float)

    out = model(style_img.reshape(1, 3, 256, 256))
    print(len(out.items()))
    
if __name__ == '__main__':
    main()