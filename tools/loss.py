import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange


def norm(x):
    return x * 2 - 1

def de_norm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G
        
    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))
        return content_loss, style_loss
    

def compute_gram(x):
    b, ch, h, w = x.size()
    f = x.view(b, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (h * w * ch)
    return G


class LossOperator():
    def __init__(self, args):
        self.args = args
        self.real_label = torch.tensor(1.0)
        self.fake_label = torch.tensor(0.0)
        self.criterion_L1 = torch.nn.L1Loss().to(args.device)
        self.criterion_L2 = torch.nn.MSELoss().to(args.device)
        self.vgg = Vgg19().to(self.args.device)
        self.vgg_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

        filter_x = [[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]]
        filter_y = [[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]]
        filter_diag1 = [[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]]
        filter_diag2 = [[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]]
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2

        weight_array = torch.cuda.FloatTensor(weight_array).permute(3,2,0,1)
        self.weight = nn.Parameter(data=weight_array, requires_grad=False)

    def calc_vgg_loss(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        style_loss = 0.0
        for i in list(range(len(x_vgg))):
            content_loss += self.vgg_weights[i] * self.criterion_L1(x_vgg[i], y_vgg[i].detach())
            style_loss += self.criterion_L1(compute_gram(x_vgg[i]), compute_gram(y_vgg[i].detach()))
        return content_loss, style_loss
    
    def calc_total_variation_loss(self, mask):
        tv_h = mask[:, :, 1:, :] - mask[:, :, :-1, :]
        tv_w = mask[:, :, :, 1:] - mask[:, :, :, :-1]
        return torch.mean(torch.abs(tv_h)) + torch.mean(torch.abs(tv_w))
    
    def calc_laplacian_loss(self, flow):
        flow_x, flow_y = torch.split(flow, 1, dim=1)
        delta_x = F.conv2d(flow_x, self.weight)
        delta_y = F.conv2d(flow_y, self.weight)

        b, c, h, w = delta_x.shape
        image_elements = b * c * h * w

        loss_flow_x = (delta_x.pow(2)+ self.args.epsilon ** 2).pow(0.45)
        loss_flow_y = (delta_y.pow(2)+ self.args.epsilon ** 2).pow(0.45)

        loss_flow_x = torch.sum(loss_flow_x) / image_elements
        loss_flow_y = torch.sum(loss_flow_y) / image_elements
        return loss_flow_x + loss_flow_y