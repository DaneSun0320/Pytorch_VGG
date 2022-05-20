#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @IDE          :PyCharm
# @Project      :VGG
# @FileName     :train
# @CreatTime    :2022/5/15 10:50 
# @CreatUser    :DaneSun
import os

import torch.cuda
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import VGG

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 7
EPOCH = 20

transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
}

data_path = './dataset/'

train_data = ImageFolder(root=data_path+'train',
                         transform=transform['train'])
val_data = ImageFolder(root=data_path+'val',
                       transform=transform['val'])
train_num = len(train_data)
val_num = len(val_data)

classes = train_data.classes

train_loader = DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
validate_loader = DataLoader(dataset=val_data,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

model = VGG('vgg16',class_num = 5,init_weight=True).to(DEVICE)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)


print("Device:{}".format(DEVICE))

best_acc = 0

print("Start training...")
for epoch in range(EPOCH):
    epoch_acc = 0.0
    train_loss = 0.0
    model.train()
    for step,data in enumerate(train_loader):
        images,labels = data
        optimizer.zero_grad()
        outputs = model(images.to(DEVICE))
        loss = loss_func(outputs,labels.to(DEVICE))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        rate = (step + 1) / len(train_loader)
        a = "▉" * int(rate * 100)
        b = ' ' * int((1 - rate) * 100)
        print("\rEpoch {} [{}{}] {:.2f}% loss:{:.3f}".format(epoch + 1,a,b,rate * 100,loss.item()),end='')

    train_loss = train_loss / train_num
    print()
    model.eval()
    with torch.no_grad():
        for images,labels in validate_loader:
            outputs = model(images.to(DEVICE))
            predict_y = torch.max(outputs,dim=1)[1]
            epoch_acc += (predict_y == labels.to(DEVICE)).sum().item()
        epoch_acc = epoch_acc / val_num
        print("EPOCH {} [train loss:{:.3f},accuracy:{:.2f}%]".format(epoch + 1,train_loss,epoch_acc*100))
    if epoch_acc > best_acc:
        torch.save(model.state_dict(),'./model/VGG.pth')
        best_acc = epoch_acc
print("Finished training...")
