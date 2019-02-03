import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
import time
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    # 如果GPU可用就使用GPU
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 生成一个CNN网络
    cnn = CNN()
    
    # 加载CIFAR10训练数据
    train_image, train_label, test_image, test_label = dataset.load_CIFAR10()
    train_image = torch.from_numpy(train_image).float()
    train_label = torch.from_numpy(train_label).long()
    test_image = torch.from_numpy(test_image).float()
    
    sample_num = train_image.shape[0] # 得到训练样本的个数
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
    
    minibatch_size = 100 # 指定minibatch的大小
    minibatch_num = int(sample_num/minibatch_size)
    ave_loss = 0 # 平均损失
    cnn.to(device) # 将整个模型推送到GPU
    for epoch in range(50):  # 轮询全体训练数据的次数
        minibatch_loss_list = []
        for minibatch_idx in range(minibatch_num):
            # 加载训练数据并推送到GPU
            minibatch_train_image = train_image[(minibatch_idx * minibatch_size):((minibatch_idx + 1) * minibatch_size), :, :, :]
            minibatch_train_label = train_label[(minibatch_idx * minibatch_size):((minibatch_idx + 1) * minibatch_size)]
            minibatch_train_image = minibatch_train_image.to(device)
            minibatch_train_label = minibatch_train_label.to(device)
    
            optimizer.zero_grad() # 梯度buffer置零
    
            outputs = cnn(minibatch_train_image) # 输入训练数据，求取网络输出
            loss = criterion(outputs, minibatch_train_label) # 求取目标函数
            minibatch_loss_list.append(loss.item())
            if minibatch_idx == minibatch_num-1:
                ave_loss = sum(minibatch_loss_list) / len(minibatch_loss_list)
                print("epoch:%5d, aveloss:%10.8f" % (epoch, ave_loss))
            loss.backward() # 反向传播
            optimizer.step() # 更新权值
    
    end_time = time.time()
    print(f'Finished Training. Time cost = {end_time - start_time}')

    with torch.no_grad():
        test_image = test_image.to(device)
        predict = cnn(test_image)

    predict = predict.to("cpu")
    predict = predict.numpy()
    predict = np.argmax(predict,axis=1)
    error_rate = np.sum((predict != test_label) + 0.0) / len(test_label)
    print(f"error_rate = {error_rate}")