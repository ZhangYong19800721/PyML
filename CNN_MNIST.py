import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import io
import dataset
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from Dataset_MNIST import *


class CNN(nn.Module):
    """
        卷积神经网络
    本代码参考https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html进行修改
    """

    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    start_time = time.time()

    # 如果GPU可用就使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cnn = CNN()  # 初始化一个CNN的实例

    # 加载MNIST训练数据
    myTransform = transforms.Compose([Reshape((28,28)),Border((32,32)),ToTensor()]) # 数据预处理
    trainDataSet = Dataset_Train_MNIST('./data/mnist.mat',myTransform) # 加载训练数据集
    trainDataLoader = DataLoader(trainDataSet, batch_size=100) # minibatch的大小为100

    # 目标函数CrossEntropy
    criterion = nn.CrossEntropyLoss()

    # 准备最优化算法
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum = 0.9)

    cnn.to(device)
    batch_loss = []
    for epoch in range(30): # 对全部的训练数据进行n次遍历
        for batch_id, batch in enumerate(trainDataLoader):
            train_images = batch['image'].to(device)
            train_labels = batch['label'].to(device)

            optimizer.zero_grad()  # zero the gradient buffers
            output_data = cnn(train_images)
            loss = criterion(output_data, train_labels)
            batch_loss.append(loss.item())
            while len(batch_loss) > len(trainDataLoader):
                batch_loss.pop(0)
            ave_loss = sum(batch_loss) / len(batch_loss)
            print("epoch:%5d, batch_id:%5d, aveloss:%10.8f" % (epoch, batch_id, ave_loss))
            loss.backward()
            optimizer.step()  # Does the update

    end_time = time.time()
    print(f'train_time = {end_time - start_time}s')

    # 加载MNIST测试数据
    testDataSet = Dataset_Test_MNIST('./data/mnist.mat',myTransform) # 加载测试数据集
    testDataLoader = DataLoader(testDataSet, batch_size=100) # minibatch的大小为100
    error_count = 0
    with torch.no_grad():
        for batch_id, batch in enumerate(testDataLoader):
            test_images = batch['image'].to(device)
            test_lables = batch['label'].numpy()
            predict = cnn(test_images).to('cpu').numpy()
            predict = np.argmax(predict, axis=1)
            error_count += np.sum((predict != test_lables) + 0.0)

    error_rate = error_count / len(testDataSet)
    print(f"error_rate = {error_rate}")
