import torch
import torch.nn as nn
#from torchsummary import summary

num_classes = 10
feature_fields = [12, 24, 48, 48, 48, 48, 96, 96, 96, 112, 192]    

class ConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, padding: int, stride: int, bias: bool):
        super(ConvBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias
        
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(out_dim)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    

class GeneralCNN(nn.Module):
    def __init__(self, num_classes: int =num_classes, feature_fields: list[int] = feature_fields):
        super(GeneralCNN, self).__init__()
        self.num_classes = num_classes
        self.feature_fields = feature_fields
        
        self.block1 = ConvBlock(3, feature_fields[0], kernel_size=3, padding=2, stride=2, bias=False)
        self.block2 = ConvBlock(feature_fields[0], feature_fields[1], kernel_size=3, padding=1, stride=1, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block3 = ConvBlock(feature_fields[1], feature_fields[2], kernel_size=3, padding=1, stride=1, bias=False)
        self.block4 = ConvBlock(feature_fields[2], feature_fields[3], kernel_size=3, padding=1, stride=1, bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block5 = ConvBlock(feature_fields[3], feature_fields[4], kernel_size=3, padding=1, stride=1, bias=False)
        self.block6 = ConvBlock(feature_fields[4], feature_fields[5], kernel_size=3, padding=1, stride=1, bias=False)
        self.block7 = ConvBlock(feature_fields[5], feature_fields[6], kernel_size=3, padding=1, stride=1, bias=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block8 = ConvBlock(feature_fields[6], feature_fields[7], kernel_size=3, padding=1, stride=1, bias=False)
        self.block9 = ConvBlock(feature_fields[7], feature_fields[8], kernel_size=3, padding=1, stride=1, bias=False)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block10 = ConvBlock(feature_fields[8], feature_fields[9], kernel_size=3, padding=1, stride=1, bias=False)
        self.block11 = ConvBlock(feature_fields[9], feature_fields[10], kernel_size=3, padding=1, stride=1, bias=False)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fully_net = torch.nn.Sequential(
            nn.Linear(192*16, 64),
            torch.nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            nn.ELU(inplace=True),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.pool3(x)
        
        x = self.block8(x)
        x = self.block9(x)
        x = self.pool4(x)
        
        x = self.block10(x)
        x = self.block11(x)
        x = self.pool5(x)
        
        x = x.view(x.size(0), -1)
        x = self.fully_net(x)
    
        return x
    

def load_CNN():
    model = GeneralCNN()
    return model
        
if __name__ == '__main__':
    model = load_CNN()
    model.eval()
    
    summary(model, (3, 255, 255))