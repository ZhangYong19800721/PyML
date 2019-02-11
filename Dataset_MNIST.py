# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class Reshape(object):
    """
            变形变换
        MNIST手写数字图片从mat文件中读入格式是784（=28x28）行的列向量，数据类型uint8，本类
        将其变换为28x28大小的矩阵（灰度图片）
    """
    def __init__(self,output_size):
        assert isinstance(output_size,tuple)
        self.output_size = output_size
        
    def __call__(self,sample):
        image,label = sample['image'], sample['label']
        image = image.reshape(self.output_size)
        return {'image':image, 'label':label}

class Border(object):
    """
            加边框变换
        MNIST手写数字图片的原始大小是28x28，加上边框之后变为32x32，目的是为了方便后续做卷积
        操作
    """
    
    def __init__(self,output_size):
        assert isinstance(output_size,tuple)
        self.output_size = output_size
        
    def __call__(self,sample):
        image,label = sample['image'], sample['label']
        new_image = np.zeros(self.output_size,dtype=np.uint8)
        new_image[2:30,2:30] = image
        return {'image':new_image, 'label':label}
    
class ToTensor(object):
    """
            转换为张量
        将数据样本变换为张量，方便在GPU上计算
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        new_image = (image / 255.0).reshape((1,32,32))
        new_image = torch.from_numpy(new_image).float()
        new_label = torch.tensor(label).long()
        return {'image': new_image, 'label': new_label}
    
class ToDevice(object):
    """
            将数据推送至GPU
    """
    def __init__(self,device):
        self.device = device

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        return {'image': image.to(self.device), 'label': label.to(self.device)}

class Dataset_Train_MNIST(Dataset):
    """ 
            MNIST数据集
        手写数字图片数据集，图片大小为28*28的灰度图，包含数字0~9。
        训练集共6万张图片，测试集共1万张图片
    """

    def __init__(self, filemat, transform=None):
        """
        参数:
            filemat (string): mat文件的全路径名称 
            transform (callable, optional): 应用在数据样本上的预处理操作
        """
        self.mnist = sio.loadmat(filemat) # 从mat文件中加载mnist数据
        self.transform = transform

    def __len__(self):
        return self.mnist['mnist_train_images'].shape[1]

    def __getitem__(self, idx):
        image = self.mnist['mnist_train_images'][:,idx]
        label = self.mnist['mnist_train_labels'][idx,0]
        sample = {'image':image,'label':label}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class Dataset_Test_MNIST(Dataset):
    """ 
            MNIST测试集
        手写数字图片数据集，图片大小为28*28的灰度图，包含数字0~9。
        测试集共1万张图片
    """

    def __init__(self, filemat, transform=None):
        """
        参数:
            filemat (string): mat文件的全路径名称 
            transform (callable, optional): 应用在数据样本上的预处理操作
        """
        self.mnist = sio.loadmat(filemat) # 从mat文件中加载mnist数据
        self.transform = transform

    def __len__(self):
        return self.mnist['mnist_test_images'].shape[1]

    def __getitem__(self, idx):
        image = self.mnist['mnist_test_images'][:,idx]
        label = self.mnist['mnist_test_labels'][idx,0]
        sample = {'image':image,'label':label}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
if __name__ == '__main__':
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    myTransform = transforms.Compose([Reshape((28,28)),Border((32,32)),ToTensor()])
    dataset = Dataset_Train_MNIST('./data/mnist.mat',myTransform)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=100)
    for i_batch, i_sample_batch in enumerate(dataloader):
        print(i_batch)
        plt.subplot(2,2,1)
        plt.imshow(i_sample_batch['image'][0][0].numpy())
        plt.subplot(2,2,2)
        plt.imshow(i_sample_batch['image'][1][0].numpy())
        plt.subplot(2,2,3)
        plt.imshow(i_sample_batch['image'][2][0].numpy())
        plt.subplot(2,2,4)
        plt.imshow(i_sample_batch['image'][3][0].numpy())
        plt.show()