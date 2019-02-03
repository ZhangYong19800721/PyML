

import matplotlib.pyplot as plt


if __name__ == '__main__':
    file = "D:\workspace\PyML\data\cifar-10-python\cifar-10-batches-py\data_batch_1"
    data = unpickle(file)
    n = 14
    x = data[b'data'][n]
    x = x.reshape((32,32,3),order='F').transpose((1,0,2))
    plt.imshow(x)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(classes[data[b'labels'][n]])