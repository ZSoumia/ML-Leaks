from torch import nn
class Classifier(nn.Module):
    """
        Implementation of the basline Mnist classifier 
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        self.fc_input_size = 800 if in_channels == 3 else 32*4*4
        self.fc1 = nn.Linear(in_features=self.fc_input_size, out_features=512) 
        self.fc2 = nn.Linear(in_features= 512,out_features=out_channels)
  
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avg_pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.avg_pool(x)

        x = x.view(x.size(0), -1) # flatten in order to feed to the fc layer 
 
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.softmax(x)
        return x
