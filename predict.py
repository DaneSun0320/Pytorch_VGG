#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @IDE          :PyCharm
# @Project      :VGG
# @FileName     :predict
# @CreatTime    :2022/5/20 11:47 
# @CreatUser    :DaneSun


import torch
from PIL import Image
from torchvision import transforms

from model import VGG

transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ]
)

classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
img = Image.open('./dataset/val/daisy/169371301_d9b91a2a42.jpg') # 加载图片 [height, width, channel]
img = transform(img) # 将图片转换成tensor [channel, height, width]
img = torch.unsqueeze(img, 0) # 增加一个维度 [batch, channel, height, width]

net = VGG("vgg16",class_num=5)
net.load_state_dict(torch.load('./model/VGG.pth'))
net.eval()
with torch.no_grad():
    output = torch.squeeze(net(img))
    predict = torch.softmax(output, 0)
    predict_class = torch.argmax(predict).numpy()
    #   计算概率
print(classes[int(predict_class)],predict[predict_class].item())