from typing import Type, Union, List
import torch
from escnn import nn, gspaces


class E2BasicBlock(torch.nn.Module):
    expansion: int = 1
    def __init__(self, r2_act, inplanes, planes, stride, downsample, norm_layer ):
        super().__init__()
        in_type = nn.FieldType(r2_act, inplanes * [r2_act.regular_repr])
        out_type = nn.FieldType(r2_act, planes * [r2_act.regular_repr])

        self.conv1 = nn.R2Conv(
            in_type, out_type, 3, padding=1, stride=stride, bias=False
        )
        self.bn1 = norm_layer(out_type)
        self.relu = nn.ReLU(out_type, True)
        self.conv2 = nn.R2Conv(out_type, out_type, 3, padding=1, stride=1, bias=False)
        self.bn2 = norm_layer(out_type)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class E2BottleNeck(torch.nn.Module):
    expansion: int = 2
    def __init__(self, r2_act, inplanes, planes, stride, downsample, norm_layer , base_width=64):
        super().__init__()
        in_type = nn.FieldType(r2_act, inplanes * [r2_act.regular_repr])
        out_type = nn.FieldType(r2_act, planes * [r2_act.regular_repr])
        exp_type = nn.FieldType(r2_act, self.expansion * planes * [r2_act.regular_repr])

        self.conv1 = nn.R2Conv(in_type, out_type, kernel_size=1, bias=False)
        self.bn1 = norm_layer(out_type)
        self.relu1 = nn.ReLU(out_type, True)

        self.conv2 = nn.R2Conv(
            out_type, out_type, kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn2 = norm_layer(out_type)
        self.relu2 = nn.ReLU(out_type, True)

        self.conv3 = nn.R2Conv(out_type, exp_type, kernel_size=1, bias=False)
        self.bn3 = norm_layer(exp_type)
        self.relu3 = nn.ReLU(exp_type, True)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


class E2ResNet(torch.nn.Module):
    def __init__(
        self,
        r2_act: gspaces.GSpace2D,
        block: Type[Union[E2BasicBlock, E2BottleNeck]],
        layers: List[int],
        num_classes: int,
        base_width: int,
    ):
        super().__init__()

        self.r2_act = r2_act
        self.in_type = nn.FieldType(r2_act, 3 * [r2_act.trivial_repr])

        self.norm_layer = nn.InnerBatchNorm
        self.dilation = 1
        self.base_width = base_width
        self.inplanes = self.base_width

        out_type = nn.FieldType(r2_act, self.base_width * [r2_act.regular_repr])

        self.mask_module = nn.MaskModule(self.in_type, 224, margin=1)
        self.conv1 = nn.R2Conv(
            self.in_type,
            out_type,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = self.norm_layer(out_type)
        self.maxpool = nn.PointwiseMaxPool(
            out_type, kernel_size=3, stride=2, padding=1
        )

        self.relu1 = nn.ReLU(out_type, True)
        self.layer1 = self._make_layer(r2_act, block, base_width, layers[0])
        self.layer2 = self._make_layer(r2_act, block, base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(r2_act, block, base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(r2_act, block, base_width * 8, layers[3], stride=2)

        out_type = nn.FieldType(r2_act, block.expansion * base_width * 8 * [r2_act.regular_repr])
        self.avgpool = nn.PointwiseAdaptiveAvgPool(out_type, (1, 1))

        self.gpool = nn.GroupPooling(out_type)
        self.fc = torch.nn.Linear(block.expansion * base_width * 8, num_classes)

    def _make_layer(
        self,
        r2_act: gspaces.GSpace,
        block: Type[Union[E2BasicBlock, E2BottleNeck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> torch.nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes:
            in_type = nn.FieldType(r2_act, self.inplanes * [r2_act.regular_repr])
            out_type = nn.FieldType(r2_act, block.expansion * planes * [r2_act.regular_repr])
            downsample = nn.SequentialModule(
                nn.R2Conv(in_type, out_type, kernel_size=1, padding=0, stride=stride),
                self.norm_layer(out_type),
            )

        layers = []
        layers.append(
            block(
                r2_act, self.inplanes, planes, stride, downsample, self.norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(r2_act, self.inplanes, planes, 1, None, self.norm_layer)
            )

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = nn.GeometricTensor(x, self.in_type)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.gpool(x).tensor
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class E2ResNet50(torch.nn.Module):
    def __init__(self,
        r2_act : gspaces.GSpace2D,
        layers: List[int],
        num_classes: int = 10,
        base_width: int = 64,
        ):
        super().__init__()

        self.r2_act = r2_act
        self.num_classes = num_classes
        self.inplanes = base_width
        self.norm_layer = nn.InnerBatchNorm
    
        self.in_type = nn.FieldType(r2_act, 3 * [r2_act.trivial_repr])
        out_type = nn.FieldType(r2_act, self.inplanes * [r2_act.regular_repr])

        self.mask_module = nn.MaskModule(self.in_type, 256, margin=1)
        self.conv1 = nn.R2Conv(self.in_type, out_type,kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_layer(out_type)
        self.relu1 = nn.ReLU(out_type, True)
        self.maxpool = nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(E2BottleNeck, 64, layers[0])
        self.layer2 = self._make_layer(E2BottleNeck,  128, layers[1], stride=2)
        self.layer3 = self._make_layer(E2BottleNeck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(E2BottleNeck, 512, layers[3], stride=2)
        
        out_type = nn.FieldType(r2_act,  2 * 512 * [r2_act.regular_repr])

        self.avgpool = nn.PointwiseAdaptiveAvgPool(out_type, (1, 1))

        self.gpool = nn.GroupPooling(out_type)
        
        self.fc = torch.nn.Linear(512*2, num_classes) 
    

        
    
    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            in_type = nn.FieldType(self.r2_act, self.inplanes * [self.r2_act.regular_repr])
            out_type = nn.FieldType(self.r2_act, planes * block.expansion * [self.r2_act.regular_repr])
            downsample = nn.SequentialModule(
                nn.R2Conv(in_type, out_type, kernel_size=1, stride=stride, padding=0),
                self.norm_layer(out_type),
            )
        
        layers = []
        layers.append(
            block(
                self.r2_act, self.inplanes, planes, stride, downsample, self.norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.r2_act, self.inplanes, planes, 1, None, self.norm_layer)
            )

        return torch.nn.Sequential(*layers) 

    
    
    def forward(self, x):
        x = nn.GeometricTensor(x, self.in_type)
        x = self.mask_module(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.gpool(x)
        x = x.tensor
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def c1resnet18(num_classes: int=10, base_width: int=80):
    r2_act = gspaces.trivialOnR2()
    return E2ResNet(
        r2_act,
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        base_width=base_width,
    )

def d1resnet18(num_classes: int=10, base_width: int=54):
    r2_act = gspaces.flip2dOnR2()
    return E2ResNet(
        r2_act,
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        base_width=base_width,
    )

def c4resnet18(num_classes: int=10, base_width: int=40):
    r2_act = gspaces.rot2dOnR2(N=4)
    return E2ResNet(
        r2_act,
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        base_width=54,
    )

def d4resnet18(num_classes: int=10, base_width: int=28):
    r2_act = gspaces.flipRot2dOnR2(N=4)
    return E2ResNet(
        r2_act,
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        base_width=54,
    )

def small_c4resnet(num_classes: int=10):
    r2_act = gspaces.rot2dOnR2(N=4)
    return E2ResNet(
        r2_act,
        block=E2BasicBlock,
        layers=[1, 1, 1, 1],
        num_classes=num_classes,
        base_width=5
    )

def small_c4resnet50():
    r2_act = gspaces.rot2dOnR2(N=4)
    return E2ResNet50(r2_act,layers=[3, 2, 2, 2])

def c4resnet50():
    r2_act = gspaces.rot2dOnR2(N=4)
    return E2ResNet50(r2_act,layers=[3, 4, 6, 3])

def c2resnet50():
    r2_act = gspaces.rot2dOnR2(N=2)
    return E2ResNet50(r2_act,layers=[3, 2, 2, 2])

def d4resnet50():
    r2_act = gspaces.flipRot2dOnR2(N=4)
    return E2ResNet50(r2_act,layers=[3, 2, 2, 2])