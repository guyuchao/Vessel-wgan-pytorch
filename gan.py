from torch import nn
from torchvision import models
import torch.nn.functional as F
import torch


class block(nn.Module):
    def __init__(self,in_filters,n_filters):
        super(block,self).__init__()
        self.deconv1 = nn.Sequential(
            nn.Conv2d(in_filters, n_filters, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU())
    def forward(self, x):
        x=self.deconv1(x)
        return x

class generator(nn.Module):
    # initializers
    def __init__(self, n_filters=32):
        super(generator, self).__init__()
        self.down1=nn.Sequential(
            block(3,n_filters),
            block(n_filters,n_filters),
            nn.MaxPool2d((2,2)))
        self.down2 = nn.Sequential(
            block(n_filters, 2*n_filters),
            block(2*n_filters, 2*n_filters),
            nn.MaxPool2d((2, 2)))
        self.down3 = nn.Sequential(
            block(2*n_filters, 4*n_filters),
            block(4*n_filters, 4*n_filters),
            nn.MaxPool2d((2, 2)))
        self.down4 = nn.Sequential(
            block(4*n_filters, 8 * n_filters),
            block(8 * n_filters, 8 * n_filters),
            nn.MaxPool2d((2, 2)))
        self.down5 = nn.Sequential(
            block(8 * n_filters, 16 * n_filters),
            block(16 * n_filters, 16 * n_filters))

        self.up1=nn.Sequential(
            block(16 * n_filters+8*n_filters, 8 * n_filters),
            block(8 * n_filters, 8 * n_filters))
        self.up2 = nn.Sequential(
            block(8 * n_filters+4*n_filters, 4 * n_filters),
            block(4 * n_filters, 4 * n_filters))
        self.up3 = nn.Sequential(
            block(4 * n_filters+2*n_filters,2 * n_filters),
            block(2 * n_filters, 2 * n_filters))
        self.up4 = nn.Sequential(
            block(2 * n_filters+n_filters,  n_filters),
            block( n_filters,  n_filters))

        self.out=nn.Sequential(
            nn.Conv2d(n_filters,1,kernel_size=1)
        )
    # forward method
    def forward(self, x):
        #print(x.size())
        x1=self.down1(x)
        #print(x1.size())
        x2=self.down2(x1)
        #print(x2.size())
        x3 = self.down3(x2)
        #print(x3.size())
        x4 = self.down4(x3)
        #print(x4.size())
        x5 = self.down5(x4)
        #print(x5.size())
        x = self.up1(F.upsample(torch.cat((x4,x5),dim=1),scale_factor=2))
        x = self.up2(F.upsample(torch.cat((x, x3), dim=1), scale_factor=2))
        x = self.up3(F.upsample(torch.cat((x, x2), dim=1), scale_factor=2))
        x = self.up4(F.upsample(torch.cat((x, x1), dim=1), scale_factor=2))
        x=F.sigmoid(self.out(x))
        return x#b,1,w,h

class discriminator(nn.Module):
    def __init__(self,n_filters):
        super(discriminator,self).__init__()
        self.down1 = nn.Sequential(
            block(4, n_filters),
            block(n_filters, n_filters),
            nn.MaxPool2d((2, 2)))
        self.down2 = nn.Sequential(
            block(n_filters, 2 * n_filters),
            block(2 * n_filters, 2 * n_filters),
            nn.MaxPool2d((2, 2)))
        self.down3 = nn.Sequential(
            block(2 * n_filters, 4 * n_filters),
            block(4 * n_filters, 4 * n_filters),
            nn.MaxPool2d((2, 2)))
        self.down4 = nn.Sequential(
            block(4 * n_filters, 8 * n_filters),
            block(8 * n_filters, 8 * n_filters),
            nn.MaxPool2d((2, 2)))
        self.down5 = nn.Sequential(
            block(8 * n_filters, 16 * n_filters),
            block(16 * n_filters, 16 * n_filters))
        self.out = nn.Linear(16*n_filters,1)
    def forward(self, x):
        x=self.down1(x)
        #print(x.size())
        x = self.down2(x)
        #print(x.size())
        x = self.down3(x)
        #print(x.size())
        x = self.down4(x)
        #print(x.size())
        x = self.down5(x)
        #print(x.size())
        x=F.avg_pool2d(x, kernel_size=x.size()[2:]).view(x.size()[0], -1)


        x=self.out(x)
        x = F.sigmoid(x)
        #print(x.size())
        return x#b,1


if __name__=='__main__':
    from torch.nn.functional import Variable
    D=discriminator(32).cuda()
    t=Variable(torch.ones((2,4,512,512)).cuda())
    print(D(t).size())