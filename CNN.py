import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import io
import utility
import numpy as np


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
    cnn = CNN()  # 初始化一个CNN的实例

    # 加载MNIST训练数据
    train_datas, train_label, test_datas, test_label = utility.load_mnist()
    train_image = train_datas.reshape((60000, 1, 28, 28))
    mnist_train_image = np.zeros((60000, 1, 32, 32))
    mnist_train_image[:,:,2:30,2:30] = train_image
    test_image = test_datas.reshape((10000, 1, 28, 28))
    mnist_test_image = np.zeros((10000, 1, 32, 32))
    mnist_test_image[:,:,2:30,2:30] = test_image


    train_datas = torch.randn(10, 1, 32, 32)
    output_data = cnn(train_datas)
    print(output_data[0])

    target = torch.randn(10)  # a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()

    loss = criterion(output_data, target)
    print(loss)

    cnn.zero_grad()
    loss.backward()

    print("cnn.conv1.bias.grad = ", cnn.conv1.bias.grad)

    # create optimizer
    learn_rate = 0.01
    optimizer = optim.SGD(cnn.parameters(), lr=learn_rate)

    # in your training loop:
    optimizer.zero_grad()  # zero the gradient buffers
    output_data = cnn(input_data)
    loss = criterion(output_data, target)
    loss.backward()
    optimizer.step()  # Does the update
