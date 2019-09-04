from torch.utils.data import *
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time
import random as rand
import sys
import os

'''

通过实验得到的一些结论：
    1.Dropout效果不好(严重影响到了最终的loss(750左右->100以内))
    2.BN、GN效果不好,IN收敛速度极慢
    3.有relu比没有relu效果更好
    4.采用residual的机制效果更好，更加容易收敛
    5.第一层采用多维度的特征提取方式会提升效果,dstack效果比线性叠加更好
    6.类似于WDSR的trick，浅层采用较深的feature map，深层采用较浅的feature map效果会更好
    7.设置不同的学习率对于模型的收敛影响较大，是否是因为训练样本的特征维度太少了，所以导致训练的过程十分的不稳定

model:BN
model1:No Normalization
model2:GN
model3::IN
model4:no dropout 1->32(3)->16(3)->(16(3)+16(1))->(1+1)->1
model5:no relu(x)
model6:1->128(3)->128(1)->(16,16)->(1+1)->1
model7:1->(128(3),128(1))->128(1)->(16,16)->(1+1)->1
model8:1->(cat(128(3),128(1))->256)->128(1)->(16,16)->(1+1)->1  (SOTA)  [lr step=5时，每5步出现一次比较大的性能提升，因此有必要对其每一个epoch进行学习率的指数衰减]
model9:nn.Linear()
model10:weight normalization
'''
class trainDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_size = imgSize
        self.train_data = []
        self.allFilePath = []
        for i,dir in enumerate(os.listdir(self.img_dir)):
            list2 = os.listdir(os.path.join(self.img_dir,dir))
            for j in list2:
                self.allFilePath.append(os.path.join(self.img_dir,dir,j))
        for file in self.allFilePath:
            f = open(file)
            for idx,line in enumerate(f):
                # if idx>10:break
                data = line.strip().split(',')
                list_data = [np.float64(i) for i in data]
                list_data.append(0.0)
                numpy_data = np.array(list_data)
                # print(list_data)
                numpy_data=numpy_data.reshape(1,3,3)
                # print(numpy_data)
                self.train_data.append(numpy_data)

        self.is_transform = is_transform
        self.train_data = np.array(self.train_data)[:int(len(self.train_data)*0.8)]#train:test=8:1
        print("train_data.shape:",self.train_data.shape)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        train_x = self.train_data[index]
        train_y = self.train_data[index]
        raw_data = np.array(train_x)
        #将其中的4个通道的数据置0
        raw_data[0][0][1]=raw_data[0][1][0]=raw_data[0][1][2]=raw_data[0][2][1]=0
        train_x=raw_data
        return train_x,train_y

class testDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_size = imgSize
        self.test_data = []
        self.allFilePath = []
        for i,dir in enumerate(os.listdir(self.img_dir)):
            list2 = os.listdir(os.path.join(self.img_dir,dir))
            for j in list2:
                self.allFilePath.append(os.path.join(self.img_dir,dir,j))
        for file in self.allFilePath:
            f = open(file)
            for idx,line in enumerate(f):
                # if idx>10:break
                data = line.strip().split(',')
                list_data = [np.float64(i) for i in data]
                list_data.append(0.0)
                numpy_data = np.array(list_data)
                # print(list_data)
                numpy_data=numpy_data.reshape(1,3,3)
                # print(numpy_data)
                self.test_data.append(numpy_data)

        self.is_transform = is_transform
        self.test_data = np.array(self.test_data)[int(len(self.test_data)*0.8):]
        print("test_data.shape:",self.test_data.shape)

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        train_x = self.test_data[index]
        train_y = self.test_data[index]
        raw_data = np.array(train_x)
        #将其中的4个通道的数据置0
        raw_data[0][0][1]=raw_data[0][1][0]=raw_data[0][1][2]=raw_data[0][2][1]=0
        train_x=raw_data
        return train_x,train_y

class Baseline(nn.Module):
    """docstring for ClassName"""
    def __init__(self):#网络中涉及到需要进行更新的参数需要在此处进行声明
        super(Baseline, self).__init__()
        self.hidden11 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3,padding=(1,1),stride=(1,1))),# (size-kernel_size+1)/stride
            # nn.InstanceNorm2d(num_features=32),
            # nn.GroupNorm(num_groups=8,num_channels=32),
            # nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.hidden12 = nn.Sequential(

            nn.utils.weight_norm(nn.Conv2d(in_channels=1, out_channels=128, kernel_size=1,stride=(1,1))),#为啥kernel_size=3不padding也能跑。而且loss降不下去
            # nn.InstanceNorm2d(num_features=32),
            # nn.GroupNorm(num_groups=4,num_channels=16),
            # nn.BatchNorm2d(num_features=16),
            # nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1,stride=(1,1))),#为啥kernel_size=3不padding也能跑。而且loss降不下去
            # nn.InstanceNorm2d(num_features=32),
            # nn.GroupNorm(num_groups=4,num_channels=16),
            # nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.hidden31 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(in_channels=128, out_channels=16, kernel_size=3,padding=(1,1),stride=(1,1))),# (size-kernel_size+1)/stride
            nn.ReLU(),
        )
        self.hidden32 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(in_channels=128, out_channels=16, kernel_size=1, stride=(1,1))),
            nn.ReLU(),
        )
        self.hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=(1,1)),
        )
        # self.Linear = nn.Sequential(
        #     nn.Linear(9,128),
        #     nn.Linear(128,64),
        #     nn.Linear(64,9)
        # )
    def forward(self, x):
        # x = x.view(x.size(0),-1)
        # x = self.Linear(x)
        x11 = self.hidden11(x)
        x12 = self.hidden12(x)
        x1=torch.cat((x11,x12),1);
        x2 = self.hidden2(x1)
        x3 = self.hidden31(x2)+self.hidden32(x2)
        x4 = self.hidden4(x3)
        x =  x4+x
        return x

def train_model(model, optimizer, num_epochs=25):

    for epoch in range(num_epochs):
        print("epoch:",epoch)
        lossAver = []
        model.train(True)
        lrScheduler.step()
        start = time.time()
        lossAvg = []
        for i,(input,label) in enumerate(trainloader):
            # if i> 5:break
            # print((input,label))
            # print("1:type(input)--",type(input))
            X = Variable(input.float())
            
            # label=label.view(label.size(0),-1)

            Y = Variable(label.float(),requires_grad=False)
            pre_x = model(X)
            # loss = torch.log(criteria(pre_x,Y))#定义损失函数
            loss = criteria(pre_x,Y)
            optimizer.zero_grad()#反向传播（梯度清零，避免求导所得的梯度进行累加）
            loss.backward()#方向传播求各个张量的梯度
            optimizer.step()#更新参数
            lossAvg.append(loss)
            if(i%50==0):
                print("[epoch:{}][{}/{}]Loss is:{}".format(epoch,i,trainloader.__len__(),sum(lossAvg)/len(lossAvg)))
            if(i%500==0):
                loss_train.append((sum(lossAvg)/len(lossAvg)).item())
        model.eval()
        MSE,spendtime=eval(model,testDirs,epoch)
        loss_val.append(MSE)
        torch.save(model.state_dict(), storeName +'-'+str(MSE)+'.pth')
        print("[epoch:{}] the Mean Square Error is:{} spend {} second".format(epoch,MSE,spendtime))
        print("loss_train:",loss_train)
        print("loss_val:",loss_val)

def eval(model,test_dirs,Epoch):
    f = open("predict.txt","a")
    f.write("Epoch:"+str(Epoch)+"\n")
    start = time.time()
    MSEAvg=[]
    for i,(input,label) in enumerate(testloader):
        # if i> 5:break
        

        # label=label.view(label.size(0),-1)
        
        x = Variable(input.float())
        pred_x = model(x)
        pre = pred_x.data.numpy()
        label = label.numpy()
        MSEAvg.append(np.sum((pre-label)**2)/9)
        torch.set_printoptions(precision=3)#设置打印的精度
        f.write("prediction:"+str(pre.flatten())+"  "+"label："+str(label.flatten())+"Mse="+str(np.sum((pre-label)**2)/len(pre))+"\n")
        # print(criterior(pred_x,label))
        # MSEAvg.append()
    spendtime = time.time()-start
    f.close()
    return sum(MSEAvg)/len(MSEAvg),spendtime

epochs =100
batchSize = 8
trainDirs=testDirs = "./subject_2"
storeName = './model10/model-mse'
imgSize = (3,3)

model = Baseline()

loss_train = []
loss_val=[]

print(model)

load_model = ""#./model/model-mse-298014.06794835324.pth./model8/model-mse-6222.341514192763.pth
if os.path.exists(load_model):
    print("Loading the exists model:{}".format(load_model))
    model.load_state_dict(torch.load(load_model))
else:
    print("The model is not exists!")


optimizer_conv = optim.Adam(model.parameters())#
# optimizer_conv=optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
lrScheduler = lr_scheduler.ExponentialLR(optimizer_conv,gamma=0.631)#lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)
criteria = nn.L1Loss()
traindata = trainDataLoader(trainDirs,imgSize)
trainloader = DataLoader(traindata, batch_size=batchSize, shuffle=True, num_workers=0)

testdata = testDataLoader(testDirs,imgSize)
testloader = DataLoader(testdata, batch_size=1, shuffle=False, num_workers=0)

#f_loss = open("train_loss.txt","a")

model_conv = train_model(model, optimizer_conv, num_epochs=epochs)

print("loss_train:",loss_train)
print("loss_val:",loss_val)