'''
This code is based on https://github.com/zijundeng/pytorch-semantic-segmentation which is licensed under the MIT License.
Copyright (c) 2017 ZijunDeng

DESCRIPTION:     Python script for SegNet
Date: 20.10.2019

For details on the license please have a look at MasterThesis/Licenses/MIT_License.txt
'''

import torch
from torch import nn
from torchvision import models
from pjval_ml import PJVAL_SHARE
import os
from ..utils import initialize_weights


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(_DecoderBlock, self).__init__()
        middle_channels = in_channels // 2
        layers = [
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        ]
        layers += [
                      nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
                      nn.BatchNorm2d(middle_channels),
                      nn.ReLU(inplace=True),
                  ] * (num_conv_layers - 2)
        layers += [
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class SegNet(nn.Module):
    #print("SegNet(nn.Module):")
    def __init__(self, num_classes, pretrained=True):
        super(SegNet, self).__init__()

        class VGG(nn.Module):

            def __init__(self, features, num_classes=1000, init_weights=True):
                super(VGG, self).__init__()
                self.features = features
                self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
                self.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, num_classes),
                )
                if init_weights:
                    self._initialize_weights()

            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x

            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, 0.01)
                        nn.init.constant_(m.bias, 0)

        def make_layers(cfg, batch_norm=False):
            layers = []
            in_channels = 3
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = v
            return nn.Sequential(*layers)

        cfgs = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                  'M'],
        }

        def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
            if pretrained:
                kwargs['init_weights'] = False
            model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
            if pretrained:
                # state_dict = load_state_dict_from_url(model_urls[arch],
                #                                  progress=progress)
                # state_dict = r"D:\All_data\resynthesis\DetectingTheUnexpected_weights\vgg19_bn-c79401a0.pth"
                # state_dict = torch.load("D:/All_data/resynthesis/DetectingTheUnexpected_weights/vgg19_bn-c79401a0.pth")
                state_dict = torch.load(os.path.join(PJVAL_SHARE, 'data', 'samples_osr_yolo', 'GAN', 'dataset',
                                                     'Weights', 'vgg19_bn-c79401a0.pth'))

                model.load_state_dict(state_dict)
                print("loaded vgg19_bn-c79401a0.pth")
            return model

        def vgg19_bn(pretrained=False, progress=True, **kwargs):
            return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
        #vgg = models.vgg19_bn(pretrained=pretrained)
        vgg = vgg19_bn(pretrained=pretrained)


        features = list(vgg.features.children())
        self.enc1 = nn.Sequential(*features[0:7])
        self.enc2 = nn.Sequential(*features[7:14])
        self.enc3 = nn.Sequential(*features[14:27])
        self.enc4 = nn.Sequential(*features[27:40])
        self.enc5 = nn.Sequential(*features[40:])

        self.dec5 = nn.Sequential(
            *([nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)] +
              [nn.Conv2d(512, 512, kernel_size=3, padding=1),
               nn.BatchNorm2d(512),
               nn.ReLU(inplace=True)] * 4)
        )
        self.dec4 = _DecoderBlock(1024, 256, 4)
        self.dec3 = _DecoderBlock(512, 128, 4)
        self.dec2 = _DecoderBlock(256, 64, 2)
        self.dec1 = _DecoderBlock(128, num_classes, 2)
        initialize_weights(self.dec5, self.dec4, self.dec3, self.dec2, self.dec1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec5 = self.dec5(enc5)
        dec4 = self.dec4(torch.cat([enc4, dec5], 1))
        dec3 = self.dec3(torch.cat([enc3, dec4], 1))
        dec2 = self.dec2(torch.cat([enc2, dec3], 1))
        dec1 = self.dec1(torch.cat([enc1, dec2], 1))
        return dec1


