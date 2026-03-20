import torch
import torch.nn.functional as F

def compute_anomaly_map(ft_list, fs_list, upsample_size):
    maps = []
    for ft, fs in zip(ft_list, fs_list):
        ft_hat = ft / (ft.norm(dim=1, keepdim=True) + 1e-10)
        fs_hat = fs / (fs.norm(dim=1, keepdim=True) + 1e-10)
        diff = 0.5 * ((ft_hat - fs_hat)**2).sum(dim=1, keepdim=True)
        diff_up = F.interpolate(diff, size=upsample_size, mode='bilinear', align_corners=False)
        maps.append(diff_up)
    anomaly_map = maps[0] * maps[1] * maps[2]
    return anomaly_map
