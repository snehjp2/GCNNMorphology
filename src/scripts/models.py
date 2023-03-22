import torch
import torch.nn as nn
from torch.nn import functional as F
from e2cnn import gspaces
from e2cnn import nn as e2cnn_nn
from e2wrn import WRNc8c4c1, WRNd8d4d1


num_classes = 10

if torch.cuda.is_available():
    device = torch.device('gpu')
else:
    device = torch.device('cpu')

class GeneralSteerableCNN(torch.nn.Module):
    
    def __init__(self, N, n_classes=num_classes, reflections = False):
        
        super(GeneralSteerableCNN, self).__init__()
        
        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.N = N

        if (reflections == True) and (self.N == 1):
          self.r2_act = gspaces.Flip2dOnR2()

        elif reflections == True:
          self.r2_act = gspaces.FlipRot2dOnR2(N=self.N)

        else:
          self.r2_act = gspaces.Rot2dOnR2(N=self.N)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = e2cnn_nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the regular representation of C8
        out_type = e2cnn_nn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.block1 = e2cnn_nn.SequentialModule(
            e2cnn_nn.MaskModule(in_type, 256, margin=1),
            e2cnn_nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            e2cnn_nn.InnerBatchNorm(out_type),
            e2cnn_nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = e2cnn_nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block2 = e2cnn_nn.SequentialModule(
            e2cnn_nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2cnn_nn.InnerBatchNorm(out_type),
            e2cnn_nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = e2cnn_nn.SequentialModule(
            e2cnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = e2cnn_nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block3 = e2cnn_nn.SequentialModule(
            e2cnn_nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2cnn_nn.InnerBatchNorm(out_type),
            e2cnn_nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = e2cnn_nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block4 = e2cnn_nn.SequentialModule(
            e2cnn_nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2cnn_nn.InnerBatchNorm(out_type),
            e2cnn_nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = e2cnn_nn.SequentialModule(
            e2cnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields
        out_type = e2cnn_nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block5 = e2cnn_nn.SequentialModule(
            e2cnn_nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2cnn_nn.InnerBatchNorm(out_type),
            e2cnn_nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = e2cnn_nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block6 = e2cnn_nn.SequentialModule(
            e2cnn_nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            e2cnn_nn.InnerBatchNorm(out_type),
            e2cnn_nn.ReLU(out_type, inplace=True)
        )
        self.pool3 = e2cnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
        
        self.gpool = e2cnn_nn.GroupPooling(out_type)
        
        # number of output channels
        c = self.gpool.out_type.size
        
        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(c, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, n_classes),
        )
    
    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = e2cnn_nn.GeometricTensor(input, self.input_type)
        
        # apply each equivariant block
        
        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        
        x = self.block5(x)
        x = self.block6(x)
        
        # pool over the spatial dimensions
        x = self.pool3(x)
        
        # pool over the group
        x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor
        
        # classify with the final fully connected layers)
        # use NLL loss
        x = self.fully_net(x.reshape(x.shape[0], -1))
        x = F.log_softmax(x, dim=-1)
        
        return x
    

D2_model = GeneralSteerableCNN(N=2,reflections=True).to(device)
D4_model = GeneralSteerableCNN(N=4,reflections=True).to(device)
D8_model = GeneralSteerableCNN(N=8,reflections=True).to(device)
D16_model = GeneralSteerableCNN(N=16,reflections=True).to(device)
C2_model = GeneralSteerableCNN(N=2).to(device)
C4_model = GeneralSteerableCNN(N=4).to(device)
C8_model = GeneralSteerableCNN(N=8).to(device)
C16_model = GeneralSteerableCNN(N=16).to(device)
WRNd8d4d1 = WRNd8d4d1.to(device)
WRNc8c4c1 = WRNc8c4c1.to(device)

# load WRN-50-2:
WRN_50_2 = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
for param in WRN_50_2.parameters():
    param.requires_grad = False
WRN_50_2.fc = nn.Linear(WRN_50_2.fc.in_features, num_classes)

# load Resnet18:
RN_18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
for param in RN_18.parameters():
    param.requires_grad = False
RN_18.fc = nn.Linear(RN_18.fc.in_features, num_classes)

if __name__ == "__main__":
    
    ### input size = (batch_size, 3, 256, 256)
    input_size = (3, 256, 256)
    
    models = [D2_model, D4_model, D8_model, D16_model, C2_model, C4_model, C8_model, C16_model, WRN_50_2, RN_18, WRNc8c4c1, WRNd8d4d1]
    
    for model in models:
        print(model)
