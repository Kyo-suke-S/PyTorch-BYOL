import torchvision.models as models
import torch
from models.mlp_head import MLPHead


class ResNet18(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet18, self).__init__()
        if kwargs['name'] == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif kwargs['name'] == 'resnet50':
            resnet = models.resnet50(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)
    
    
class ResNet_withAttention(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet_withAttention, self).__init__()
        if kwargs['name'] == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif kwargs['name'] == 'resnet50':
            resnet = models.resnet50(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-2])
        #self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])
        self.in_features = resnet.fc.in_features

        self.backbone_avg_pool = resnet.avgpool
        self.ext_avg_pool = torch.nn.AdaptiveAvgPool2d(kwargs['encoded_img_size'])

    def forward(self, x):
        pre_h = self.encoder(x)
        h = self.backbone_avg_pool(pre_h)
        h = h.view(h.shape[0], h.shape[1])
        ext_H = self.ext_avg_pool(pre_h)
        return h, ext_H
