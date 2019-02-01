import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import io
import utility
import numpy as np
import time
import sys


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
    # 如果GPU可用就使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cnn = CNN()  # 初始化一个CNN的实例

    # 加载MNIST训练数据
    train_datas, train_label, test_datas, test_label = utility.load_mnist()

    # 准备训练图片
    train_datas = train_datas.reshape((60000, 1, 28, 28))
    mnist_train_image = np.zeros((60000, 1, 32, 32))
    mnist_train_image[:,:,2:30,2:30] = train_datas
    mnist_train_image = torch.from_numpy(mnist_train_image).float()

    # 准备测试图片
    test_datas = test_datas.reshape((10000, 1, 28, 28))
    mnist_test_image = np.zeros((10000, 1, 32, 32))
    mnist_test_image[:,:,2:30,2:30] = test_datas
    mnist_test_image = torch.from_numpy(mnist_test_image).float()

    # 准备训练标签
    mnist_train_label = torch.from_numpy(train_label).float()

    # 目标函数MSE
    criterion = nn.MSELoss()

    # 准备最优化算法
    learn_rate = 0.01 # 学习速度
    optimizer = optim.SGD(cnn.parameters(), lr=learn_rate, momentum = 0.9)
    minibatch_size = 1000

    start_time = time.time()

    minibatchNum = int(60000/minibatch_size)
    ave_loss = 0

    cnn.to(device)
    for epoch in range(30): # 对全部的训练数据进行n次学习
        minibatch_loss_list = []
        for minibatch_idx in range(minibatchNum):
            minibatch_train_image = mnist_train_image[(minibatch_idx * minibatch_size):((minibatch_idx + 1) * minibatch_size), :, :, :]
            minibatch_train_label = mnist_train_label[(minibatch_idx * minibatch_size):((minibatch_idx + 1) * minibatch_size), :]
            minibatch_train_image = minibatch_train_image.to(device)
            minibatch_train_label = minibatch_train_label.to(device)

            optimizer.zero_grad()  # zero the gradient buffers
            output_data = cnn(minibatch_train_image)
            loss = criterion(output_data, minibatch_train_label)
            minibatch_loss_list.append(loss.item())
            if minibatch_idx == minibatchNum-1:
                ave_loss = sum(minibatch_loss_list) / len(minibatch_loss_list)
                print("epoch:%5d, aveloss:%10.8f" % (epoch, ave_loss))
            loss.backward()
            optimizer.step()  # Does the update

    end_time = time.time()
    print(f'time_cost = {end_time - start_time}s')

    with torch.no_grad():
        mnist_test_image = mnist_test_image.to(device)
        predict = cnn(mnist_test_image)

    predict = predict.to("cpu")
    predict = predict.numpy()
    predict = np.argmax(predict,axis=1)
    error_rate = np.sum((predict != test_label) + 0.0) / len(test_label)
    print(f"error_rate = {error_rate}")
