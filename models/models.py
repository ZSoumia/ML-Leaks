from torch import nn
class Mnist_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)

        self.fc1 = nn.Linear(in_features=4*4*50, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool(x)
        x = self.tanh(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, 4*4*50) # flattening 
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class Cifar10_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool(x)
        x = self.tanh(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5) # flattening
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

"""
On CIFAR datasets, we train a standard convolutional neuralnetwork (CNN) with two convolution
 and max pooling layersplus a fully connected layer of size128and aSoftMaxlayer.We  useTanhas 
  the  activation  function.  We  set  the  learningrate  to0.001,  the  learning  rate  decay  
  to1e−07,  and  themaximum epochs of training to100.
"""