from torchvision.models import ResNet
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import tifffile
import SimpleITK as sitk


def conv3x3x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 1), stride=stride,
                     padding=(1, 1, 0), bias=False)


def conv3x3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3), stride=stride,
                     padding=1, bias=False)


class PlainBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PlainBlock, self).__init__()
        self.conv1 = conv3x3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x1(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CubicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CubicBlock, self).__init__()
        if stride < 0:
            self.conv1 = nn.ConvTranspose3d(inplanes, planes, kernel_size=(3, 3, 3), stride=-stride, bias=False)
            self.upsample = True
        else:
            self.conv1 = conv3x3x3(inplanes, planes, stride)
            self.upsample = False
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            if self.upsample:
                out = out[:,:,0:out.size()[2]-1,0:out.size()[3]-1,0:out.size()[4]-1]

        out += residual
        out = self.relu(out)

        return out


class VoxelBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(VoxelBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 1, stride)
        #self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(inplanes, planes, 1, stride)
        #self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):

    def __init__(self, num_classes=1000):
        self.inplanes = 16
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))
        self.layer1 = self._make_layer(CubicBlock, 32, 2)
        self.layer2 = self._make_layer(CubicBlock, 64, 2, stride=2)
        self.layer3 = self._make_layer(CubicBlock, 128, 4, stride=2)
        self.layer4 = self._make_layer(CubicBlock, 256, 8, stride=2)
        #self.layer5 = self._make_layer(CubicBlock, 256, 2, stride=2)
        #self.avgpool = nn.AvgPool3d((4, 6, 3),)
        self.top_conv = nn.Conv3d(256, 256, (3, 3, 3), stride=2, padding=(1, 1, 1))
        self.top = nn.Conv3d(256, num_classes, (1, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.layer5(x)

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        x = self.top_conv(x)
        x = self.relu(x)
        x = self.top(x)

        return x

    def get_top_filter_value(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #t = self.layer5(x)

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        t = self.top_conv(x)
        x = self.relu(t)
        x = self.top(x)

        return x, t


from VISoR_Brain.utils.elastix_files import get_sitk_transform_from_file
import random

class patch_data_sequential(torch.utils.data.Dataset):
    def __init__(self, image_file, spacing=16):
        self.image_file = image_file
        self.spacing = spacing
        self.init = False
        size = sitk.ReadImage(image_file[0]).GetSize()
        size = [size[0], size[1], len(image_file)]
        self.size = [(size[i] - 32) // self.spacing for i in range(3)]
        self.length = self.size[0] * self.size[1] * self.size[2]

    def _real_init(self):
        self.image = sitk.ReadImage(self.image_file)

    def __getitem__(self, item: int):
        if not self.init:
            self._real_init()
            self.init = True
        pos = {}
        pos[2] = item // (self.size[0] * self.size[1])
        pos[1] = (item % (self.size[0] * self.size[1])) // self.size[0]
        pos[0] = item % self.size[0]
        pos = [pos[i] * self.spacing + 16 for i in range(3)]
        patch = self.image[pos[0] - 16: pos[0] + 16,
                pos[1] - 16: pos[1] + 16,
                pos[2] - 16: pos[2] + 16]
        patch = sitk.GetArrayFromImage(patch)
        patch = np.float32([patch])
        return patch

    def __len__(self):
        return self.length


class patch_data(torch.utils.data.Dataset):
    def __init__(self, image_file, label_file, transform_file):
        self.image_file = image_file
        self.label_file = label_file
        self.transform_file = transform_file
        self.init = False

    def _real_init(self):
        transform = sitk.ReadImage(self.transform_file)
        transform = sitk.Cast(transform, sitk.sitkVectorFloat64)
        self.transform = sitk.DisplacementFieldTransform(transform)
        self.image = sitk.ReadImage(self.image_file)
        label_image = sitk.ReadImage(self.label_file)
        self.label_image = sitk.GetArrayFromImage(label_image)
        brain_map = sitk.ReadImage('C:/Users/chaoyu/Documents/projects/cfos_counting/annotation_25.nrrd')
        brain_map = sitk.GetArrayFromImage(brain_map)
        self.brain_map = np.transpose(brain_map, [2, 1, 0])

    def __getitem__(self, item):
        if not self.init:
            self._real_init()
            self.init = True
        while 1:
            map_pos = [random.randint(0, self.brain_map.shape[2 - i] - 1) for i in range(3)]
            label = self.brain_map[map_pos[2], map_pos[1], map_pos[0]]
            #if self.brain_map[map_pos[2], map_pos[1], map_pos[0]] >= 0:
            #    break
            if label == 672:
                label = 0
                break
            if label == 632:
                label = 1
                break
            if label == 961:
                label = 2
                break
            if label in {329, 981, 201, 1047, 1070, 1038, 1062}:
                label = 3
                break
            continue
        brain_pos = self.transform.TransformPoint(map_pos)
        brain_pos = [brain_pos[i] * 6.25 for i in range(3)]
        brain_pos = [min(brain_pos[i], self.image.GetSize()[i] - 16) for i in range(3)]
        brain_pos = [int(max(brain_pos[i], 16)) for i in range(3)]
        patch = self.image[brain_pos[0] - 16: brain_pos[0] + 16,
                brain_pos[1] - 16: brain_pos[1] + 16,
                brain_pos[2] - 16: brain_pos[2] + 16]
        patch = sitk.GetArrayFromImage(patch)
        patch = np.float32([patch])
        return patch, np.int64([[[label]]])

    def __len__(self):
        return 100000000


if __name__ == '__main__':
    image_path = 'F:/TEST_DATA/Mouse_Brain/20180914_ZMN_WH_438_1_1/Reconstruction/Brain'
    image_files = [os.path.join(image_path, 'Z{:05d}_C1.tif'.format(i)) for i in range(1000, 2000)]
    data = patch_data(image_files, 'F:/chaoyu/test/thy1/intensity.tif', 'F:/chaoyu/test/thy1/438/deformationField.mhd')
    for i in range(1000):
        p, l = data[i]
        print(l)
        tifffile.imwrite('F:/chaoyu/test/thy1/patches/{}_{}.tif'.format(i, l), p)
