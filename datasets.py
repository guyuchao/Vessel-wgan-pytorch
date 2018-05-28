import os
import os.path as osp
import numpy as np
from PIL import Image
from gycutils.gycaug import ColorAug,Add_Gaussion_noise,Random_horizontal_flip,Random_vertical_flip,Compose_imglabel,Random_crop
import collections
import torch
import torchvision
from transform import ReLabel, ToLabel, Scale
from torch.utils import data
from transform import HorizontalFlip, VerticalFlip
from torchvision.transforms import Compose
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor

def default_loader(path):
    return Image.open(path)

class VOCDataSet(data.Dataset):
    def __init__(self, root, split="train", img_transform=None, label_transform=None,image_label_transform=None):
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.image_label_transform=image_label_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()

        data_dir = osp.join(root, "eyedata",split)
        # for split in ["train", "trainval", "val"]:
        imgsets_dir = osp.join(data_dir,  "img")
        for name in os.listdir(imgsets_dir):
            name = os.path.splitext(name)[0]
            img_file = osp.join(data_dir, "img/%s.tif" % name)
            label_file = osp.join(data_dir, "label/%s.gif" % name)
            self.files[split].append({
                "img": img_file,
                "label": label_file
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')
        # img = img.resize((256, 256), Image.NEAREST)
        # img = np.array(img, dtype=np.uint8)

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")
        #label_size = label.size
        # label image has categorical value, not continuous, so we have to
        # use NEAREST not BILINEAR
        # label = label.resize((256, 256), Image.NEAREST)
        # label = np.array(label, dtype=np.uint8)
        # label[label == 255] = 21

        if self.image_label_transform is not None:
            img,label=self.image_label_transform(img,label)

        if self.img_transform is not None:
            imgs= self.img_transform(img)
            # img_h = self.img_transform(self.h_flip(img))
            # img_v = self.img_transform(self.v_flip(img))

            #imgs = [img_o]

        #else:
            #imgs = img

        if self.label_transform is not None:
            labels= self.label_transform(label)
            # label_h = self.label_transform(self.h_flip(label))
            # label_v = self.label_transform(self.v_flip(label))
            #labels = [label_o]
        #else:
            #labels = label
        return imgs, labels

if __name__ == '__main__':

    input_transform = Compose([
        ColorAug(),
        Add_Gaussion_noise(prob=0.5),
        #Scale((512, 512), Image.BILINEAR),
        ToTensor(),
        Normalize([.485, .456, .406], [.229, .224, .225]),

    ])
    target_transform = Compose([
        #Scale((512, 512), Image.NEAREST),
        #ToSP(512),
        ToLabel(),
        ReLabel(255, 1),
    ])

    img_label_transform = Compose_imglabel([
        Random_crop(512,512),
        Random_horizontal_flip(0.5),
        Random_vertical_flip(0.5),
    ])
    dst = VOCDataSet("./", img_transform=input_transform,label_transform=target_transform,image_label_transform=img_label_transform)
    trainloader = data.DataLoader(dst, batch_size=1)

    for i, data in enumerate(trainloader):
        imgs, labels = data



