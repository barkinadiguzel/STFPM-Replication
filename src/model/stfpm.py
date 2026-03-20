import torch
from backbone.teacher import Teacher
from backbone.student import Student
from modules.loss import feature_loss
from modules.anomaly_map import compute_anomaly_map

class STFPM:
    def __init__(self, device="cuda"):
        self.device = device
        self.teacher = Teacher().to(device)
        self.student = Student().to(device)

    def train_step(self, x):
        x = x.to(self.device)
        ft_list = self.teacher(x)
        fs_list = self.student(x)
        loss = feature_loss(ft_list, fs_list)
        return loss

    @torch.no_grad()
    def infer(self, x):
        x = x.to(self.device)
        H, W = x.shape[2], x.shape[3]
        ft_list = self.teacher(x)
        fs_list = self.student(x)
        anomaly_map = compute_anomaly_map(ft_list, fs_list, upsample_size=(H, W))
        return anomaly_map
