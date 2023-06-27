import torch
import torch.nn as nn
from torch.nn import functional as F
from escnn import gspaces
from escnn import nn as escnn_nn
import e2resnet 
import e2wrn
import torchvision
#from torchsummary import summary

num_classes = 10
feature_fields = [24, 24, 48, 48, 48, 48, 96, 96, 96]    

class GeneralSteerableCNN(torch.nn.Module):
    
    def __init__(self, N, n_classes=num_classes, feature_fields = feature_fields, reflections = False):
        
        super(GeneralSteerableCNN, self).__init__()
        
        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.N = N

        if (reflections == True) and (self.N == 1):
          self.r2_act = gspaces.flip2dOnR2()

        elif reflections == True:
          self.r2_act = gspaces.flipRot2dOnR2(N=self.N)

        else:
          self.r2_act = gspaces.rot2dOnR2(N=self.N)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = escnn_nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the regular representation of C8
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[0]*[self.r2_act.regular_repr])
        
        
        self.block1 = escnn_nn.SequentialModule(
            escnn_nn.MaskModule(in_type, 256, margin=1),
            escnn_nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False, stride=2),
            escnn_nn.InnerBatchNorm(out_type),
            escnn_nn.ReLU(out_type, inplace=True)
        )
        
        self.pool1 = escnn_nn.SequentialModule(
            escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[1]*[self.r2_act.regular_repr])
        self.block2 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False, stride=1),
            escnn_nn.InnerBatchNorm(out_type),
            escnn_nn.ReLU(out_type, inplace=True)
        )

        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[2]*[self.r2_act.regular_repr])
        self.block3 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False, stride=2),
            escnn_nn.InnerBatchNorm(out_type),
            escnn_nn.ReLU(out_type, inplace=True)
        )
        
        self.pool2 = escnn_nn.SequentialModule(
            escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[3]*[self.r2_act.regular_repr])
        self.block4 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False, stride=1),
            escnn_nn.InnerBatchNorm(out_type),
            escnn_nn.ReLU(out_type, inplace=True)
        )
        
        in_type = self.block4.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[4]*[self.r2_act.regular_repr])
        self.block5 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False, stride=1),
            escnn_nn.InnerBatchNorm(out_type),
            escnn_nn.ReLU(out_type, inplace=True)
        )
        
        in_type = self.block5.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[5]*[self.r2_act.regular_repr])
        self.block6 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False, stride=1),
            escnn_nn.InnerBatchNorm(out_type),
            escnn_nn.ReLU(out_type, inplace=True)
        )
        
        in_type = self.block6.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[6]*[self.r2_act.regular_repr])
        self.block7 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False, stride=1),
            escnn_nn.InnerBatchNorm(out_type),
            escnn_nn.ReLU(out_type, inplace=True)
        )
        self.pool3 = escnn_nn.SequentialModule(
            escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block7.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[7]*[self.r2_act.regular_repr])
        self.block8 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False, stride=1),
            escnn_nn.InnerBatchNorm(out_type),
            escnn_nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block8.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = escnn_nn.FieldType(self.r2_act, feature_fields[8]*[self.r2_act.regular_repr])
        self.block9 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False, stride=1),
            escnn_nn.InnerBatchNorm(out_type),
            escnn_nn.ReLU(out_type, inplace=True)
        )
        self.pool4 = escnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
        
        self.gpool = escnn_nn.GroupPooling(out_type)
        
        # number of output channels
        # b, c, h, w = self.gpool.evaluate_output_shape(self.pool3.out_type)
        # d = c*h*w
        c = self.gpool.out_type.size
        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(9*c, 64),
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
        x = self.pool1(x)
        
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool2(x)
        
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.pool3(x)
        
        x = self.block8(x)
        x = self.block9(x)
        
        # pool over the spatial dimensions
        x = self.pool4(x)
        
        # pool over the group
        x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor
        
        # classify with the final fully connected layers)
        # use NLL loss
        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x


'''
D2_model = GeneralSteerableCNN(N=2,reflections=True)
D4_model = GeneralSteerableCNN(N=4,reflections=True)
D8_model = GeneralSteerableCNN(N=8,reflections=True)
D16_model = GeneralSteerableCNN(N=16,reflections=True)
C2_model = GeneralSteerableCNN(N=2)
C4_model = GeneralSteerableCNN(N=4)
C8_model = GeneralSteerableCNN(N=8)
C16_model = GeneralSteerableCNN(N=16)
WRNd8d4d1 = WRNd8d4d1
WRNc8c4c1 = WRNc8c4c1
'''

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
    # for param in WRN_50_2.parameters():
    #     param.requires_grad = False
    WRN_50_2.fc = nn.Linear(WRN_50_2.fc.in_features, num_classes)
    return WRN_50_2

def load_resnet18():
    RN_18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # for param in RN_18.parameters():
    #     param.requires_grad = False
    RN_18.fc = nn.Linear(RN_18.fc.in_features, num_classes)
    return RN_18

def load_resnet50():
    RN_50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    # for param in RN_50.parameters():
    #     param.requires_grad = False
    RN_50.fc = nn.Linear(RN_50.fc.in_features, num_classes)
    return RN_50

def load_densenet121():
    densenet121 = torchvision.models.densenet121(pretrained=True)
    # for param in densenet121.parameters():
    #     param.requires_grad = False
    densenet121.classifier = nn.Linear(densenet121.classifier.in_features, num_classes)
    return densenet121


model_dict = {
    'ResNet18' : load_resnet18, ## done
    'ResNet50' : load_resnet50, ## done
    'WRN50_2' : load_wrn50_2, ## done
    'densenet121' : load_densenet121, ## done
    'D2' : load_d2, ## done 
    'D4' : load_d4, ## done
    'D8' : load_d8, ## done
    'D16' : load_d16, ## done
    'C2' : load_c2, ## done
    'C4' : load_c4, ## done
    'C8' : load_c8,  ## done
    'C16' : load_c16, ## done
    'WRN16d8d4d1': e2wrn.wrn16_8_stl_d8d4d1,  ## done
    'WRN16d8d4d4' : e2wrn.wrn16_8_stl_d8d4d4, ## done
    'WRN16d1d1d1' : e2wrn.wrn16_8_stl_d1d1d1, ## done
    'WRN28_10_d8d4d1' : e2wrn.wrn28_10_d8d4d1, ## done
    'WRN28_7_d8d4d1' : e2wrn.wrn28_7_d8d4d1, ## done
    'WRN28_10_c8c4c1' : e2wrn.wrn28_10_c8c4c1, ## done
    'WRN28_10_d1d1d1' : e2wrn.wrn28_10_d1d1d1, ## done
    'c1resnet18' : e2resnet.c1resnet18, ## done
    'd1resnet18' : e2resnet.d1resnet18, ## done
    'c4resnet18' : e2resnet.c4resnet18, ## done
    'd4resnet18' : e2resnet.d4resnet18, ## done
    'small_c4resnet18' : e2resnet.small_c4resnet,
    'c4resnet50' : e2resnet.c4resnet50, ## done 
    'c2resnet50' : e2resnet.c2resnet50, ## done
    'small_c4resnet50' : e2resnet.small_c4resnet50,
    'd4resnet50' : e2resnet.d4resnet50
}

if __name__ == "__main__":

    ## model checks 
    
    model = model_dict['D8']()
    model.eval()
    x = torch.rand(size=(1,3,256,256))
    y = model(x)
    print(y.shape)

    print(f'Trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')