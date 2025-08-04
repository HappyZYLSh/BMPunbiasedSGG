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
        self.flag = torch.tensor(flag).squeeze()  # 扩展为[1, num_classes]

    def compute_pp(self, sample, mu, var):
        """计算多元正态分布概率密度"""
        b, feature_dim = sample.shape
        c = mu.shape[0]
        var = var + 1e-8  # 数值稳定性
        sample = sample.unsqueeze(1)  # [b, 1, feature_dim]
        mu = mu.unsqueeze(0)          # [1, c, feature_dim]
        var = var.unsqueeze(0)        # [1, c, feature_dim]
        
        exponent = torch.exp(-(torch.square(sample - mu)) / (2 * var))
        coefficient = 1 / torch.sqrt(2 * torch.pi * var)
        log_pdf_per_dim = torch.log(coefficient) + torch.log(exponent)
        joint_log_pdf = torch.sum(log_pdf_per_dim, dim=2)  # [b, c]
        return torch.exp(joint_log_pdf)  # [b, c]
    
    def KL_consist_loss(self, pp, logits):
        """计算pp分布与logits分布的KL散度"""
        P = torch.softmax(pp, dim=1)  # pp转换为概率分布
        Q = torch.softmax(logits, dim=1)  # logits转换为概率分布
        log_P = torch.log(P + 1e-10)
        log_Q = torch.log(Q + 1e-10)
        kl_div = torch.sum(P * (log_P - log_Q), dim=1)  # 每个样本的KL散度
        return kl_div.mean()  # 批次平均


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
        var = torch.exp(0.5 * var)


        for attr_name in ['flag', 'p_gammma', 'n_gammma1', 'n_gammma2', 'alpha']:
            if hasattr(self, attr_name):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    setattr(self, attr_name, attr_value.to(device))
        sigmoid_p = torch.sigmoid(logits)
        p_t = (targets * sigmoid_p) + ((1 - targets) * (1 - sigmoid_p))
        p_t = 1-p_t

        pp = self.compute_pp(features, mu, var)  # [b, c]
        # 初始化 p_g
        p_g = torch.zeros_like(p_t)  # [b, c]

        # 类别是否是多数/少数
        is_majority_class = (self.flag == 0)  # [c]
        is_minority_class = (self.flag == 1)  # [c]

        # 多数类：全部赋值为 p_gammma
        p_g[:, is_majority_class] = self.p_gammma

        # 少数类：先全部赋值为 n_gammma1
        p_g[:, is_minority_class] = self.n_gammma1

        # 找出哪些是少数类 & 同时是正样本
        # targets: [b, c] 是 one-hot
        is_positive = (targets == 1)  # [b, c]
        minority_class_positions = torch.tensor(is_minority_class).to(is_positive.device)  # [c]
        # print(is_positive.dtype)  # 应该是 torch.bool
        # print(minority_class_positions.dtype)  # 应该是 torch.bool

        # 取出少数类中的正样本：即 targets[i,j] == 1 且 self.flag[j] == 1
        is_positive_and_minority = is_positive & minority_class_positions.unsqueeze(0)  # [b, c]

        # 把这些位置赋值为 n_gammma2
        p_g[is_positive_and_minority] = self.n_gammma2


        modulating_factor = p_t ** p_g  # [b, c]
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = modulating_factor * alpha_t * ce_loss
        else:
            focal_loss = modulating_factor * ce_loss
        
        con_loss = self.KL_consist_loss(pp,logits)
        focal_loss = focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()
        # print(con_loss)
        # print(focal_loss)
        return focal_loss,con_loss  
    