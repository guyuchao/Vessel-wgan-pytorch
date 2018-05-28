import torch
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
from torch.autograd import Variable
from transform import Colorize
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
from transform import Scale
from duchdc import ResNetDUCHDC
from PIL import Image
from datasets2 import VOCDataSet
from PIL import Image
from duchdc import ResNetDUCHDC
import numpy as np
from tqdm import tqdm
from transform import ToLabel,ReLabel
from gycutils.gycaug import Compose_imglabel,Random_crop,Random_vertical_flip,Random_horizontal_flip
import torch
val_transform = Compose([
        ToTensor(),
        Normalize([.485, .456, .406], [.229, .224, .225]),

    ])
target_transform = Compose([
        ToLabel(),
        ReLabel(255, 1),
    ])

img_label_transform = Compose_imglabel([
        Random_crop(512,512),
        Random_horizontal_flip(0.5),
        Random_vertical_flip(0.5),
    ])


valloader = data.DataLoader(VOCDataSet("./",split='val', img_transform=val_transform,
                                         label_transform=target_transform,image_label_transform=img_label_transform),
                              batch_size=1, shuffle=False, pin_memory=True)


state_dict = torch.load("./pth/fcn-deconv-1000.pth")
model=torch.nn.DataParallel(ResNetDUCHDC(2))

model.cuda()
model.load_state_dict(state_dict)
model.eval()
import os
os.mkdir('./pth/test')

for index, (imgs,labels) in tqdm(enumerate(valloader)):
    imgs = Variable(imgs.cuda())
    outputs = model(imgs)
    outputs2 = outputs[0]
    outputs2 = F.softmax(outputs2, dim=0).cpu().data.numpy()
    np.save("./pth/test/%d.jpg"%index,outputs2)
