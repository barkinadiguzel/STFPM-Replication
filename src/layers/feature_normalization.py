import torch

def l2_normalize(feat, eps=1e-10):
    norm = torch.norm(feat, p=2, dim=1, keepdim=True) + eps
    return feat / norm
