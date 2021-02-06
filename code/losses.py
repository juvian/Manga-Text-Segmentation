from fastai.callbacks import hook_outputs
import torch.nn.functional as F
from torch import nn,tensor
from ssim import SSIM
from numpy import log10
import torch

def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)
from functools import reduce 

class FeatureLoss(nn.Module):
    def __init__(self, model, layer_ids, layer_wgts, base_loss=F.l1_loss):
        super().__init__()
        self.model = nn.Sequential(*list(model.children())[:layer_ids[-1] + 1])
        self.loss_features = [self.model[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))] + ['PSNR', 'SSIM', 'pEPs']
        self.base_loss = base_loss
        self.mse = nn.MSELoss()  
        self.ssim = SSIM(window_size=11)    

    def make_features(self, x, clone=False):
        self.model(x.cuda())
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target, *args, reduction = 'sum', **kwargs):
        #print(input.shape, target.shape)
        psnr = 10 * log10(1 / self.mse(input, target))
        ssim = self.ssim(input, target)
        pEPs = (input - target).mul(255).abs().le(0.1).sum(dim=1).eq(3).sum().div(input.numel()/input.shape[1]).mul(100)

        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        
        feat_losses = [self.base_loss(input,target)]
        feat_losses += [self.base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        feat_losses += [self.base_loss(gram_matrix(f_in), gram_matrix(f_out))*w* 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        
        self.metrics = dict(zip(self.metric_names, feat_losses + [psnr, ssim, pEPs]))
        
        if reduction == 'mean':
            return torch.mean(torch.stack(feat_losses))
        elif reduction == 'sum':
            return sum(feat_losses)
        else:
            raise reduction   
    
    def __del__(self): self.hooks.remove()
    

#https://www.kaggle.com/iafoss/unet34-dice-0-87    
def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0

    iflat = input.view(-1).float()
    tflat = target.view(-1).float()
    intersection = (iflat * tflat).sum()
    
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma, average = True):
        super().__init__()
        self.gamma = gamma
        self.average = average
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss 

        if self.average:        
            return loss.mean()
        else:
            return loss.sum()    
    
class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma, normalize = True):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        self.normalize = normalize
        
    def forward(self, input, target):
        if self.normalize:
            target = target.clone()
            target[target >= 3] -= 3
            target = target != 0
        loss = self.alpha*self.focal(input, target.float()) - torch.log(dice_loss(input, target.float()))
        return loss.mean()   

class BCELoss(nn.Module):
    def __init__(self, text_weight = 1, normalize = True):
        super().__init__()
        self.normalize = normalize     
        self.text_weight = text_weight  

    def forward(self, input, target):
        if self.normalize:
            target = target.clone()
            target[target >= 3] -= 3
            target = target != 0
        
        weight = tensor([self.text_weight]).to(input.device)

        return torch.nn.BCEWithLogitsLoss(pos_weight = weight)(input, target.float())    