import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.Multivar import ExtendedMultivariateNormal as MultivariateNormal
from torch.distributions.multivariate_normal import MultivariateNormal as MultivariateNormal1
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA

class MinorityfocalLossAndConloss(nn.Module):

    def __init__(self, gamma=2,pos_gamma = 1,neg_gamma = 1, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.p_gammma = 1  # 多数类调制指数
        self.n_gammma1 = neg_gamma  # 少数类负样本调制指数
        self.n_gammma2 = pos_gamma  # 少数类正样本调制指数
        self.flag = None  # [1, num_classes]，0=多数类，1=少数类

    def set_flag(self, flag):
        self.flag = flag.unsqueeze(0)  # 扩展为[1, num_classes]

    def compute_pp(self, sample, mu, var):
        b, feature_dim = sample.shape
        c = mu.shape[0]
        var = var + 1e-8
        sample = sample.unsqueeze(1)
        mu = mu.unsqueeze(0)
        var = var.unsqueeze(0)
        exponent = torch.exp(-(torch.square(sample - mu)) / (2 * var))
        coefficient = 1 / torch.sqrt(2 * torch.pi * var)
        log_pdf_per_dim = torch.log(coefficient) + torch.log(exponent)
        joint_log_pdf = torch.sum(log_pdf_per_dim, dim=2)
        return torch.exp(joint_log_pdf)
    
    def KL_consist_loss(self,pp,logits):
        P = torch.softmax(pp,dim=1)
        Q = logits
        log_P = torch.log(P + 1e-10)      # shape: [n, c]
        kl_div_per_element = P * (log_P - torch.log(Q + 1e-10))  # shape: [n, c]
        kl_div_per_sample = torch.sum(kl_div_per_element, dim=1)  # shape: [n]
        return kl_div_per_sample



    def forward(self, logits, targets, features, mu, var):
            # =============================
        # 1. 输入类型与设备统一处理
        # =============================

        # 确保所有输入都是 Tensor
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"logits 必须是 torch.Tensor, 但现在类型是 {type(logits)}")
        if not isinstance(targets, torch.Tensor):
            raise TypeError(f"targets 必须是 torch.Tensor, 但现在类型是 {type(targets)}")
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"features 必须是 torch.Tensor, 但现在类型是 {type(features)}")
        if not isinstance(mu, torch.Tensor):
            raise TypeError(f"mu 必须是 torch.Tensor, 但现在类型是 {type(mu)}")
        if not isinstance(var, torch.Tensor):
            raise TypeError(f"var 必须是 torch.Tensor, 但现在类型是 {type(var)}")

        # 统一转移到 CUDA 设备：cuda:2
        device = logits.device
        # print(f"[INFO] 使用设备: {device}")  # 可选，调试用

        logits = logits.to(device)
        targets = targets.to(device)
        features = features.to(device)
        mu = mu.to(device)
        var = var.to(device)


        for attr_name in ['flag', 'p_gammma', 'n_gammma1', 'n_gammma2', 'alpha']:
            if hasattr(self, attr_name):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    setattr(self, attr_name, attr_value.to(device))

        sigmoid_p = torch.softmax(logits)
        p_t = 1 - ((targets * sigmoid_p) + ((1 - targets) * (1 - sigmoid_p)))
        pp = self.compute_pp(features, mu, var)

        p_g = torch.zeros_like(p_t)
        mask_majority = (self.flag == 0)
        mask_minority = (self.flag == 1)
        p_g[mask_majority.expand_as(p_g)] = self.p_gammma

        mask_minority_pos = (targets == 1) & mask_minority.expand_as(targets)
        mask_minority_neg = (targets == 0) & mask_minority.expand_as(targets)
        p_g[mask_minority_neg] = self.n_gammma1
        p_g[mask_minority_pos] = self.n_gammma2
        p_g[mask_minority.expand_as(p_g)] *= pp[mask_minority.expand_as(p_g)]

        modulating_factor = p_t ** p_g
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = modulating_factor * alpha_t * ce_loss
        else:
            focal_loss = modulating_factor * ce_loss
        
        con_loss = self.KL_consist_loss(pp,logits)
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum(),con_loss
    