import torch
import torch.nn as nn

def feature_loss(ft_list, fs_list, alpha=[1.0,1.0,1.0]):
    total_loss = 0
    for i, (ft, fs) in enumerate(zip(ft_list, fs_list)):
        ft_hat = ft / (ft.norm(dim=1, keepdim=True) + 1e-10)
        fs_hat = fs / (fs.norm(dim=1, keepdim=True) + 1e-10)
        
        loss_map = 0.5 * ((ft_hat - fs_hat)**2).sum(dim=1)  
        loss_layer = loss_map.mean()
        total_loss += alpha[i] * loss_layer
    return total_loss
