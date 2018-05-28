from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

from gycutils.trainschedule import Scheduler
from gycutils.utils import make_trainable,calc_gradient_penalty
from gan import discriminator,generator
from datasets import VOCDataSet
from torch.optim import Adam
from loss import BCE_Loss
from transform import ReLabel, ToLabel
from torchvision.transforms import Compose, Normalize, ToTensor
import tqdm
from Criterion import Criterion
from PIL import Image
import numpy as np
import os
from gycutils.gycaug import ColorAug,Random_horizontal_flip,Random_vertical_flip,Compose_imglabel,Random_crop
input_transform = Compose([
    ColorAug(),
    ToTensor(),
    Normalize([.585, .256, .136], [.229, .124, .095]),
    ])
val_transform = Compose([
    ToTensor(),
    Normalize([.585, .256, .136], [.229, .124, .095]),
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

trainloader = data.DataLoader(VOCDataSet("./", img_transform=input_transform,
                                         label_transform=target_transform,image_label_transform=img_label_transform),
                              batch_size=2, shuffle=True, pin_memory=True)
valloader = data.DataLoader(VOCDataSet("./",split='val', img_transform=val_transform,
                                         label_transform=target_transform,image_label_transform=img_label_transform),
                              batch_size=1, shuffle=False, pin_memory=True)


schedule=Scheduler(lr=1e-4,total_epoches=4000)
D=torch.nn.DataParallel(discriminator(n_filters=32)).cuda()
G=torch.nn.DataParallel(generator(n_filters=32)).cuda()
gan_loss_percent=0.03

one=torch.FloatTensor([1])
mone=one*-1
moneg=one*-1*gan_loss_percent

one=one.cuda()
mone=mone.cuda()
moneg=moneg.cuda()

loss_func=BCE_Loss()
optimizer_D=Adam(D.parameters(),lr=1e-4,betas=(0.5,0.9),eps=10e-8)
optimizer_G=Adam(G.parameters(),lr=1e-4,betas=(0.5,0.9),eps=10e-8)

for epoch in range(schedule.get_total_epoches()):

    D.train()
    G.train()
    #train D
    make_trainable(D,True)
    make_trainable(G,False)
    for idx,(real_imgs,real_labels) in tqdm.tqdm(enumerate(trainloader)):
        real_imgs=Variable(real_imgs).cuda()
        real_labels=Variable(real_labels.unsqueeze(1)).cuda()
        D.zero_grad()
        optimizer_D.zero_grad()

        real_pair = torch.cat((real_imgs, real_labels), dim=1)
        #real_pair_y=Variable(torch.ones((real_pair.size()[0],1))).cuda()
        d_real = D(real_pair)
        d_real = d_real.mean()
        d_real.backward(mone)

        fake_pair=torch.cat((real_imgs, G(real_imgs)), dim=1)
        #fake_pair_y=Variable(torch.zeros((real_pair.size()[0],1))).cuda()
        d_fake=D(fake_pair)
        d_fake=d_fake.mean()
        d_fake.backward(one)

        #d_loss=loss_func(D(real_pair),real_pair_y)+loss_func(D(fake_pair),fake_pair_y)
        #d_loss.backward()
        gradient_penalty=calc_gradient_penalty(D,real_pair.data,fake_pair.data)
        gradient_penalty.backward()

        Wasserstein_D=d_real-d_fake
        optimizer_D.step()
    #train G

    make_trainable(D,False)
    make_trainable(G,True)
    for idx,(real_imgs,real_labels) in tqdm.tqdm(enumerate(trainloader)):
        G.zero_grad()
        optimizer_G.zero_grad()
        real_imgs=Variable(real_imgs).cuda()
        real_labels=Variable(real_labels).cuda()
        pred_labels=G(real_imgs)
        Seg_Loss=loss_func(pred_labels,real_labels.unsqueeze(1))#Seg Loss
        Seg_Loss.backward(retain_graph=True)
        fake_pair=torch.cat((real_imgs,pred_labels),dim=1)
        gd_fake=D(fake_pair)
        gd_fake=gd_fake.mean()
        gd_fake.backward(moneg)
        #Gan_Loss=loss_func(D_fack,Variable(torch.ones(fake_pair.size()[0],1)).cuda())
        #g_loss=Gan_Loss*gan_loss_percent+Seg_Loss
        #g_loss.backward()
        optimizer_G.step()
    print("epoch[%d/%d] W:%f segloss%f"%(epoch,schedule.get_total_epoches(),Wasserstein_D,Seg_Loss))



    G.eval()
    D.eval()
    if epoch%500==0:
        os.mkdir('./pth/epoch%d' % epoch)
        for i_val,(real_imgs,real_labels) in enumerate(valloader):
            real_imgs = Variable(real_imgs.cuda(), volatile=True)
            real_labels = Variable(real_labels.cuda(), volatile=True)
            outputs = G(real_imgs)
            #valloss = loss_func(outputs, real_labels)

            outputs = outputs[0].data.squeeze(0).cpu().numpy()
            pred = outputs.flatten()
            label = real_labels[0].cpu().data.numpy().flatten()
           # Criterion().precision_recall('./pth/epoch%d' % epoch, i_val, label, pred)
            Image.fromarray((outputs * 255).astype(np.uint8)).save("./pth/epoch%d/%d.jpg" % (epoch, i_val))

torch.save(G.state_dict(), "./pth/G.pth")
torch.save(D.state_dict(), "./pth/D.pth")
