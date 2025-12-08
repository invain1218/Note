import torch
from torch import nn
from torchviz import make_dot
from torchinfo import summary 

class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4):
        super(Inception,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,c1,kernel_size = 1)
        self.conv2 = nn.Sequential(
                    nn.Conv2d(in_channels,c2[0],kernel_size = 1),nn.ReLU(),
                    nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1),nn.ReLU()
                )
        self.conv3 = nn.Sequential(
                    nn.Conv2d(in_channels,c3[0],kernel_size=1),nn.ReLU(),
                    nn.Conv2d(c3[0],c3[1],kernel_size=5, padding=2),nn.ReLU()
                )
        self.conv4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
                    nn.Conv2d(in_channels,c4,kernel_size = 1),nn.ReLU(),
                )
    def forward(self,X):
        return torch.cat(
                (
                self.conv1(X),
                self.conv2(X),
                self.conv3(X),
                self.conv4(X),
                ),
                dim = 1
        )

class GoogleNet(nn.Module):
    def __init__(self,in_channels,classes):
        super(GoogleNet,self).__init__()
        self.model = nn.Sequential(
            # 7x7 conv
            nn.Conv2d(in_channels,out_channels=64,kernel_size=7,stride=2,padding=3),nn.ReLU(), # [B,C,H,W]
            # 3x3 maxpooling
            nn.MaxPool2d(kernel_size=3,stride=2), # 只改变高度和宽度，不改变通道数！
            # 1x1 conv
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1),nn.ReLU(), # stride 默认为1，移动步长
            # 3x3 conv
            nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,padding=1),nn.ReLU(),
            # 3x3 maxpooling
            nn.MaxPool2d(kernel_size=3,stride=2),
            # 2xInception
            Inception(192,c1=64,c2=[96,128],c3=[16,32],c4=32),
            Inception(256,c1=128,c2=[128,192],c3=[32,96],c4=64),
            # 3x3 maxpooling
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            # 5xInception
            Inception(480,c1=192,c2=[96,208],c3=[16,48],c4=64),
            Inception(512,c1=160,c2=[112,224],c3=[24,64],c4=64),
            Inception(512,c1=128,c2=[128,256],c3=[24,64],c4=64),
            Inception(512,c1=112,c2=[144,288],c3=[32,64],c4=64),
            Inception(528,c1=256,c2=[160,320],c3=[32,128],c4=128),
            # 3x3 maxpooling
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            # 2xInception
            Inception(832,c1=256,c2=[160,320],c3=[32,128],c4=128),
            Inception(832,c1=384,c2=[192,384],c3=[48,128],c4=128),
            # 7x7 avgpooling
            nn.AvgPool2d(kernel_size=7,stride=1),
            # 全连接层
            nn.Dropout(p=0.4), # p=40%置为0 剩下60%scale到\frac{1}{1-p}
            nn.Flatten(),
            nn.Linear(1024,classes),
            nn.Softmax(dim=1)
            )
    def forward(self,X:torch.tensor):
        for layer in self.model:
            X = layer(X)
            print(layer.__class__.__name__,'output shape:',X.shape)

X = torch.randn(size=(1,3,224,224)) # B, C, H, W
model = GoogleNet(in_channels=3,classes=1000)
print(model)
y = model(X)

summary(model, input_data=X) # 查看维度变化
import sys
import os
original_stdout = sys.stdout
output_path =os.path.join("04_DeepAR", 'model_summary.txt') 
with open(output_path, 'w') as f:
    sys.stdout = f
    summary(model, input_data=x)
    sys.stdout = original_stdout  
print(f"模型摘要已保存到 {output_path}")

def calculate_conv_hw(input_size, kernel_size, stride=1, padding=0, dilation=1):
    """计算卷积层输出尺寸"""
    output = (input_size + 2*padding - dilation*(kernel_size-1) - 1) // stride + 1
    return output

def calculate_pool_hw(input_size, kernel_size, stride=None, padding=0):
    """计算池化层输出尺寸"""
    if stride is None:
        stride = kernel_size
    output = (input_size + 2*padding - kernel_size) // stride + 1
    return output

# [B, 3, 224, 224]
# 64个7x7卷积核
calculate_conv_hw(input_size=224, kernel_size=7, stride=2, padding=3) # [B, 64, 112, 112]
# 3x3 Pooling
calculate_pool_hw(input_size=112, kernel_size=3, stride=2) # [B, 64, 55, 55]
# 64个1x1卷积核
calculate_conv_hw(input_size=64, kernel_size=1, stride=1, padding=0) # [B, 64， 55， 55]
