import torch
import torch.nn as nn
from torch.nn import functional as F
from escnn import gspaces
from escnn import nn as escnn_nn
import cnn
import e2resnet 
import torchvision
#from torchsummary import summary

num_classes = 10
feature_fields = [12, 24, 48, 48, 48, 48, 96, 96, 96, 112, 192]    

class ConvBlock(nn.Module):
    def __init__(self,in_type: escnn_nn.FieldType, out_type: escnn_nn.FieldType, kernel_size: int, padding: int, stride: int, bias: bool, MaskModule: bool = False):
        super(ConvBlock, self).__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.conv = escnn_nn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn = escnn_nn.InnerBatchNorm(out_type)
        self.act = escnn_nn.ReLU(out_type, inplace=True)
        self.MaskModule = MaskModule
        if MaskModule:
            self.mask = escnn_nn.MaskModule(in_type, 255, margin=1)
    
    def forward(self, x):
        if self.MaskModule:
            x = self.mask(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class GeneralSteerableCNN(torch.nn.Module):
    
    def __init__(self, N, n_classes=num_classes, feature_fields = feature_fields, reflections = False, maximum_frequency = None):
        
        super(GeneralSteerableCNN, self).__init__()
        
        self.N = N

        if (reflections == True) and (self.N == 1):  ## D1 case
          self.r2_act = gspaces.flip2dOnR2()

        elif reflections == True:
          self.r2_act = gspaces.flipRot2dOnR2(N=self.N)

        else:
          self.r2_act = gspaces.rot2dOnR2(N=self.N)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = escnn_nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[0]*[self.r2_act.regular_repr])
        
        self.block1 = ConvBlock(in_type, out_type, kernel_size=3, padding=2, stride=2, bias=False, MaskModule=True)

        in_type = self.block1.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[1]*[self.r2_act.regular_repr])
       
        self.block2 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)

        self.pool1 = escnn_nn.SequentialModule(
            escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        in_type = self.block2.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[2]*[self.r2_act.regular_repr])

        self.block3 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)

        in_type = self.block3.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[3]*[self.r2_act.regular_repr])

        self.block4 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)

        self.pool2 = escnn_nn.SequentialModule(
            escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        in_type = self.block4.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[4]*[self.r2_act.regular_repr])

        self.block5 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)

        in_type = self.block5.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[5]*[self.r2_act.regular_repr])
   
        self.block6 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)

        in_type = self.block6.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[6]*[self.r2_act.regular_repr])
     
        self.block7 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)

        self.pool3 = escnn_nn.SequentialModule(
            escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        in_type = self.block7.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[7]*[self.r2_act.regular_repr])
        
        self.block8 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)
        
        in_type = self.block8.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[8]*[self.r2_act.regular_repr])
        
        self.block9 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)
        self.pool4 = escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)

        in_type = self.block9.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[9]*[self.r2_act.regular_repr])

        self.block10 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)

        in_type = self.block10.out_type
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[10]*[self.r2_act.regular_repr])

        self.block11 = ConvBlock(in_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)
        self.pool5 = escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)

        self.gpool = escnn_nn.GroupPooling(out_type)
        
        # number of output channels
        # b, c, h, w = self.gpool.evaluate_output_shape(self.pool3.out_type)
        # d = c*h*w
        c = self.gpool.out_type.size
        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(25*c, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(32, n_classes),
        )
    
    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = escnn_nn.GeometricTensor(input, self.input_type)
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
        x = self.block7(x)
        x = self.pool3(x)
        
        x = self.block8(x)
        x = self.block9(x)
        x = self.pool4(x)
        
        x = self.block10(x)
        x = self.block11(x)
        x = self.pool5(x)
        # pool over the group
        x = self.gpool(x)
        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor
        # classify with the final fully connected layers)
        # use NLL loss
        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x

class SO2SteerableCNN(torch.nn.Module):

    def __init__(self, n_classes=num_classes, reflections = False):

        super(SO2SteerableCNN, self).__init__()

        # the model is equivariant under all planar rotations
        if reflections == True:
            self.r2_act = gspaces.flipRot2dOnR2(N=-1) ## O2
            
        else:
            self.r2_act = gspaces.rot2dOnR2(N=-1) ## SO2

        # the group SO(2)
        self.G = self.r2_act.fibergroup

        # the input image is a scalar field, corresponding to the trivial representation
        in_type = escnn_nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])

        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # We need to mask the input image since the corners are moved outside the grid under rotations
        self.mask = escnn_nn.MaskModule(in_type, 255, margin=1)

        # convolution 1
        # first we build the non-linear layer, which also constructs the right feature type
        # we choose 8 feature fields, each transforming under the regular representation of SO(2) up to frequency 3
        # When taking the ELU non-linearity, we sample the feature fields on N=16 points
        activation1 = escnn_nn.FourierELU(self.r2_act, 8, irreps=self.G.bl_irreps(3), N=16, inplace=True)
        out_type = activation1.in_type
        self.block1 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            escnn_nn.IIDBatchNorm2d(out_type),
            activation1,
        )

        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 16 regular feature fields
        activation2 = escnn_nn.FourierELU(self.r2_act, 16, irreps=self.G.bl_irreps(3), N=16, inplace=True)
        out_type = activation2.in_type
        self.block2 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            escnn_nn.IIDBatchNorm2d(out_type),
            activation2
        )
        # to reduce the downsampling artifacts, we use a Gaussian smoothing filter
        self.pool1 = escnn_nn.SequentialModule(
            escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 32 regular feature fields
        activation3 = escnn_nn.FourierELU(self.r2_act, 32, irreps=self.G.bl_irreps(3), N=16, inplace=True)
        out_type = activation3.in_type
        self.block3 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            escnn_nn.IIDBatchNorm2d(out_type),
            activation3
        )

        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 64 regular feature fields
        activation4 = escnn_nn.FourierELU(self.r2_act, 32, irreps=self.G.bl_irreps(3), N=16, inplace=True)
        out_type = activation4.in_type
        self.block4 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            escnn_nn.IIDBatchNorm2d(out_type),
            activation4
        )
        self.pool2 = escnn_nn.SequentialModule(
            escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields
        activation5 = escnn_nn.FourierELU(self.r2_act, 64, irreps=self.G.bl_irreps(3), N=16, inplace=True)
        out_type = activation5.in_type
        self.block5 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            escnn_nn.IIDBatchNorm2d(out_type),
            activation5
        )

        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields
        activation6 = escnn_nn.FourierELU(self.r2_act, 64, irreps=self.G.bl_irreps(3), N=16, inplace=True)
        out_type = activation6.in_type
        self.block6 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            escnn_nn.IIDBatchNorm2d(out_type),
            activation6
        )
        self.pool3 = escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

        # number of output invariant channels
        c = 64

        # last 1x1 convolution layer, which maps the regular fields to c=64 invariant scalar fields
        # this is essential to provide *invariant* features in the final classification layer
        output_invariant_type = escnn_nn.FieldType(self.r2_act, c*[self.r2_act.trivial_repr])
        self.invariant_map = escnn_nn.R2Conv(out_type, output_invariant_type, kernel_size=1, bias=False)

        # Fully Connected classifier
        self.fully_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(c),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(c, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(32, n_classes),
        )

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = self.input_type(input)

        # mask out the corners of the input image
        x = self.mask(x)

        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # Each layer outputs a new GeometricTensor, associated with the layer's output type.
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

        # extract invariant features
        x = self.invariant_map(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor

        # classify with the final fully connected layer
        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x
    
    
def load_d1():
    D1_model = GeneralSteerableCNN(N=1,reflections=True)
    return D1_model

def load_d2():
    D2_model = GeneralSteerableCNN(N=2,reflections=True)
    return D2_model

def load_d4():
    D4_model = GeneralSteerableCNN(N=4,reflections=True)
    return D4_model

def load_d8():
    D8_model = GeneralSteerableCNN(N=8,reflections=True)
    return D8_model

def load_d16():
    D16_model = GeneralSteerableCNN(N=16,reflections=True)
    return D16_model

def load_d32():
    D32_model = GeneralSteerableCNN(N=32,reflections=True)
    return D32_model

def load_d64():
    D64_model = GeneralSteerableCNN(N=64,reflections=True)
    return D64_model

def load_c1():
    C1_model = GeneralSteerableCNN(N=1)
    return C1_model

def load_c2():
    C2_model = GeneralSteerableCNN(N=2)
    return C2_model

def load_c4():
    C4_model = GeneralSteerableCNN(N=4)
    return C4_model

def load_c8():
    C8_model = GeneralSteerableCNN(N=8)
    return C8_model 

def load_c16():
    C16_model = GeneralSteerableCNN(N=16)
    return C16_model

def load_wrn50_2():
    WRN_50_2 = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
    WRN_50_2.fc = nn.Linear(WRN_50_2.fc.in_features, num_classes)
    return WRN_50_2

def load_resnet18():
    RN_18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    RN_18.fc = nn.Linear(RN_18.fc.in_features, num_classes)
    return RN_18

def load_resnet50():
    RN_50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    RN_50.fc = nn.Linear(RN_50.fc.in_features, num_classes)
    return RN_50

def load_densenet121():
    densenet121 = torchvision.models.densenet121(pretrained=True)
    densenet121.classifier = nn.Linear(densenet121.classifier.in_features, num_classes)
    return densenet121

def load_SO2():
    SO2_model = SO2SteerableCNN()
    return SO2_model

def load_O2():
    O2_model = SO2SteerableCNN(reflections=True)

model_dict = {
    'ResNet18' : load_resnet18, 
    'ResNet50' : load_resnet50, 
    'WRN50_2' : load_wrn50_2, 
    'densenet121' : load_densenet121, 
    'D1' : load_d1,
    'D2' : load_d2,  
    'D4' : load_d4, 
    'D8' : load_d8, 
    'D16' : load_d16, 
    'D32' : load_d32,
    'D64' : load_d64,
    'C1' : load_c1, 
    'C2' : load_c2, 
    'C4' : load_c4, 
    'C8' : load_c8,  
    'C16' : load_c16, 
    'CNN' : cnn.load_CNN,
    'O2' : load_O2,
    'SO2' : load_SO2,
    'c1resnet18' : e2resnet.c1resnet18, 
    'd1resnet18' : e2resnet.d1resnet18, 
    'c4resnet18' : e2resnet.c4resnet18, 
    'd4resnet18' : e2resnet.d4resnet18, 
    'small_c4resnet18' : e2resnet.small_c4resnet,
    'c4resnet50' : e2resnet.c4resnet50,  
    'c2resnet50' : e2resnet.c2resnet50, 
    'small_c4resnet50' : e2resnet.small_c4resnet50,
    'd4resnet50' : e2resnet.d4resnet50
}

if __name__ == "__main__":

    ## model checks 
    
    model = model_dict['C8']()
    model.eval()
    x = torch.rand(size=(1,3,255,255))
    y = model(x)
    print(y.shape)

    print(f'Trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')