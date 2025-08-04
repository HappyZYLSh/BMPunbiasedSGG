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
class BatchNorm(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        return self.bn(x)





# class SomeModel:
#     def __init__(self):
#         self.cov_reg = 1e-6  # 初始正则化参数

#     def adjust_cov_reg(self, selected_covs):
#         eigen_values, _ = LA.eigh(selected_covs)
#         min_eigenvalue = eigen_values.min()
#         # 根据最小特征值调整正则化参数
#         if min_eigenvalue < 1e-4:
#             self.cov_reg = 1e-6
#         elif min_eigenvalue < 1e-3:
#             self.cov_reg = 5e-4
#         else:
#             self.cov_reg = 1e-6
# class NewMultiLabelFocalLoss2(nn.Module):
#     def __init__(self, gamma_pos=1, gamma_neg=1, alpha=None, cov_reg=1e-5, min_logp=-100, reduction='mean'):
#         super(NewMultiLabelFocalLoss2, self).__init__()
#         self.gamma_pos = torch.tensor(gamma_pos, dtype=torch.float32)
#         self.gamma_neg = torch.tensor(gamma_neg, dtype=torch.float32)
#         self.alpha = alpha
#         self.reduction = reduction
#         self.cov_reg = cov_reg
#         self.min_logp = min_logp

#     def standardize_data(self, samples):
#         mean = torch.mean(samples, dim=0)
#         std = torch.std(samples, dim=0)
#         std = torch.where(std == 0, torch.tensor(1.0, dtype=std.dtype).to(device=mean.device), std)
#         standardized_samples = (samples - mean) / std
#         return standardized_samples

#     def calculate_pp(self, features, mu, log_sigma, gt_indices):
#         """
#         计算多元正态分布的概率密度函数（PDF）
#         :param features: [b, 1936], 输入特征
#         :param mu: [b, 1936], 高斯分布的均值
#         :param log_sigma: [b, 1936], 高斯分布的对数方差
#         :param gt_indices: [b], 目标类别的索引
#         :return: [b], 概率值
#         """
#         selected_mu = mu[gt_indices]  # [b, 1936]
#         selected_log_sigma = log_sigma[gt_indices]  # [b, 1936]
#         selected_sigma = torch.exp(selected_log_sigma)  # [b, 1936]

#         # 计算多元正态分布的对数概率
#         dist = MultivariateNormal(selected_mu, torch.diag_embed(selected_sigma))
#         log_pp = dist.log_prob(features)  # [b]
#         log_pp = torch.clamp(log_pp, min=self.min_logp)  # 截断最小值
#         return log_pp.exp()  # [b]

#     def forward(self, logits, targets, features, mu, log_sigma, Flag):
#         """
#         :param logits: [b, c], 模型输出的 logits
#         :param targets: [b, c], 目标标签（多标签）
#         :param features: [b, 1936], 输入特征
#         :param mu: [b, 1936], 高斯分布的均值
#         :param log_sigma: [b, 1936], 高斯分布的对数方差
#         :param Flag: [1, c], 类别标识符，Flag[i] = 0 或 1，表示第 i 类是大类或小类
#         :return: 损失值
#         """
#         device = logits.device
#         targets = targets.to(device)
#         features = features.to(device)
#         mu = mu.to(device)
#         log_sigma = log_sigma.to(device)
#         Flag = Flag.to(device)

#         # 计算二元交叉熵损失
#         ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # [b, c]

#         # 根据 Flag 设置 gamma
#         gamma = torch.ones_like(logits)  # [b, c]
#         gamma[:, Flag.squeeze() == 1] = gamma[:, Flag.squeeze() == 1] * self.gamma_pos  # 大类
#         gamma[:, Flag.squeeze() == 0] = gamma[:, Flag.squeeze() == 0] * self.gamma_neg  # 小类

#         # 计算概率 pp
#         gt_indices = targets.argmax(dim=1)  # [b], 每个样本的目标类别索引
#         pp = self.calculate_pp(features, mu, log_sigma, gt_indices)  # [b]
#         pp = pp.unsqueeze(-1)  # [b, 1]

#         # 更新 gamma
#         new_gamma = gamma * (1 - pp)  # [b, c]

#         # 计算 p_t
#         sigmoid_p = torch.sigmoid(logits)  # [b, c]
#         p_t = (targets * sigmoid_p) + ((1 - targets) * (1 - sigmoid_p))  # [b, c]

#         # 计算调制因子
#         modulating_factor = torch.pow(p_t, new_gamma)  # [b, c]

#         # 计算类别权重 alpha_t
#         if self.alpha is None:
#             class_counts = targets.sum(dim=0)  # [c]
#             total_count = class_counts.sum()  # 1
#             self.alpha = (total_count - class_counts) / total_count  # [c]
#             self.alpha = self.alpha.to(device)

#         alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)  # [b, c]

#         # 计算 Focal Loss
#         focal_loss = modulating_factor * alpha_t * ce_loss  # [b, c]

#         # 根据 reduction 参数汇总损失
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss

# class MinorityfocalLoss(nn.Module):
#     def __init__(self, gamma_pos=1, gamma_neg=1, alpha=None, cov_reg=1e-5, min_logp=-100, reduction='mean'):
#         super(MinorityfocalLoss, self).__init__()
#         self.gamma_pos = torch.tensor(gamma_pos, dtype=torch.float32)
#         self.gamma_neg = torch.tensor(gamma_neg, dtype=torch.float32)
#         self.alpha = alpha
#         self.reduction = reduction
#         self.cov_reg = cov_reg
#         self.min_logp = min_logp
#         self.model = SomeModel()

#     def standardize_data(self, samples):
#         mean = torch.mean(samples, dim = 0)
#         std = torch.std(samples, dim = 0)
#         std = torch.where(std == 0, torch.tensor(1.0, dtype = std.dtype).to(device = mean.device), std)
#         standardized_samples = (samples - mean) / std
#         return standardized_samples

#     def pca_cov_matrix(self, covariance_matrices, target_dim):
#         num_matrices, d, _ = covariance_matrices.shape
#         # 假设这里使用torch的低秩PCA方法进行近似计算
#         u, s, _ = LA.svd(covariance_matrices.view(num_matrices, -1), full_matrices=False)
#         reduced_vectors = u[:, :target_dim * target_dim] * s[:, :target_dim * target_dim]
#         reduced_matrices = reduced_vectors.view(num_matrices, target_dim, target_dim)
#         return reduced_matrices

#     def pca_mean_vectors(self, mean_vectors, target_dim):
#         u, s, _ = LA.svd(mean_vectors, full_matrices=False)
#         reduced_vectors = u[:, :target_dim] * s[:, :target_dim]
#         return reduced_vectors

#     def calculate_pp2(self, features, means, covs, gt_indices):
#         features = self.pca_mean_vectors(features, 3)
#         selected_means = means[gt_indices]
#         selected_covs = covs[gt_indices]
#         selected_means = self.pca_mean_vectors(selected_means, 3)
#         selected_covs = self.pca_cov_matrix(selected_covs, 3)

#         self.model.adjust_cov_reg(selected_covs)
#         cov_regularization = self.model.cov_reg * torch.eye(selected_covs.size(1)).unsqueeze(0).expand(
#             selected_covs.size(0), -1, -1).to(selected_covs.device)
#         selected_covs = selected_covs + cov_regularization

#         dists = MultivariateNormal1(selected_means, selected_covs)
#         log_pp = dists.log_prob(features)
#         log_pp = torch.clamp(log_pp, min=self.min_logp)
#         return log_pp.exp()

#     def calculate_pp1(self, features, means, covs, gt_indices):
#         selected_means = means[gt_indices]
#         selected_covs = covs[gt_indices]

#         dists = MultivariateNormal1(selected_means, selected_covs)
#         log_pp = dists.log_prob1(features)
#         log_pp = torch.clamp(log_pp, min=self.min_logp)
#         pp = torch.sigmoid(self.standardize_data(log_pp))
#         return pp

#     def forward(self, logits, targets, features, prototype, rel, Set):
#         device = logits.device
#         targets = targets.to(device)
#         features = features.to(device)
#         for key in prototype[rel].keys():
#             prototype[rel][key]['mean'] = prototype[rel][key]['mean'].to(device)
#             prototype[rel][key]['var'] = prototype[rel][key]['var'].to(device)

#         ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

#         gamma = torch.ones_like(logits)
#         mnos = Set[rel]["mno"]
#         gamma[:, mnos] = gamma[:, mnos] * ((1 - targets[:, mnos]) * self.gamma_neg + targets[:, mnos] * self.gamma_pos)

#         features = F.normalize(features, p=2, dim=1)
#         gt_indices = targets.argmax(dim=1)
#         means = torch.stack([prototype[rel][i]['mean'] for i in range(len(prototype[rel]))]).to(device)
#         covs = torch.stack([prototype[rel][i]['var'] for i in range(len(prototype[rel]))]).to(device)
#         pp = self.calculate_pp1(features, means, covs, gt_indices)
#         pp = pp.unsqueeze(-1)

#         new_gamma = gamma * (1-pp)

#         sigmoid_p = torch.sigmoid(logits)
#         p_t = (targets * sigmoid_p) + ((1 - targets) * (1 - sigmoid_p))

#         modulating_factor = torch.pow(p_t, new_gamma)

#         if self.alpha is None:
#             class_counts = targets.sum(dim=0)
#             total_count = class_counts.sum()
#             self.alpha = (total_count - class_counts) / total_count
#             self.alpha = self.alpha.to(device)

#         alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
#         focal_loss = modulating_factor * alpha_t * ce_loss

#         if self.reduction =='mean':
#             return focal_loss.mean()
#         elif self.reduction =='sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss

class MinorityfocalLoss(nn.Module):
    """实验组：带少数类增强的焦点损失（已修复版本）"""
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

        sigmoid_p = torch.sigmoid(logits)
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

        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()
class FocalLoss_maj(nn.Module):
    def __init__(self, alpha=None, positive_gamma = 2,negtive_gamma = 3,beta = 0.4,gamma=0,reduction = "none"):
        super(FocalLoss_maj, self).__init__()
        self.alpha = {"a":torch.ones(3),"s":torch.ones(6),"c":torch.ones(17)}
        self.omiga = {"a":torch.ones(3),"s":torch.ones(6),"c":torch.ones(17)}
        self.beta =beta
        self.gamma = gamma
        self.negtive_gamma = negtive_gamma
        self.positive_gamma = positive_gamma
        self.reduction = reduction

    def forward(self, inputs, targets, rel = None,Set = None,rel_num = None,p = None):
        """
        inputs: 模型预测的概率值，形状为 (batch_size, c)
        targets: 真实标签，形状为 (batch_size, c)，取值为0或1
        """
        # print(inputs)
        labels = F.one_hot(targets, num_classes = rel_num[rel])
        BCE_loss = labels*torch.log(inputs+1e-6)+(1-labels)*torch.log(1-inputs+1e-6)
        # print(BCE_loss)
        pt = torch.exp(BCE_loss)
        # print(pt)


        focal_loss = torch.zeros_like(inputs)
        Maj_cls = Set[rel]["maj"]
        Mno_cls = Set[rel]["mno"]
        ## 大类索引
        Maj_mask = [y in Maj_cls for y in targets ]
        Mno_mask = [y in Mno_cls for y in targets ]
        # 大类loss
        f_loss = -((1 - pt) ** self.gamma )* BCE_loss 
        focal_loss[Maj_mask,:] = f_loss[Maj_mask,:]
        # 小类loss
        print((1-pt).shape)
        print((labels*self.positive_gamma+(1-labels)*self.negtive_gamma).shape)
        print(((labels*self.positive_gamma+(1-labels)*self.negtive_gamma)*(1-p)).shape)
        print(p.shape)
        print(BCE_loss.shape)

        f_loss_1 = -1*(1 - pt) ** ((labels*self.positive_gamma+(1-labels)*self.negtive_gamma)*(1-p)) * BCE_loss
        focal_loss[Mno_mask,:] = f_loss_1[Mno_mask,:]



        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
        
    def get_alpha(self):
        return self.alpha
#########    
class MEET_loss(nn.Module):
    def __init__(self, alpha=None, positive_gamma = 2,negtive_gamma = 3,beta = 0.4,gamma=0,reduction = "none"):
        super(MEET_loss, self).__init__()

class Matching_loss(nn.Module):

    def __init__(self, alpha=None, positive_gamma = 2,negtive_gamma = 3,beta = 0.4,gamma=0,reduction = "none"):
    
        super(Matching_loss, self).__init__()


class AR_loss(nn.Module):
    def __init__(self, gamma_pos=2, gamma_neg=3, alpha=None, reduction='mean'):
        super(AR_loss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.reduction = reduction
        self.rel_num_dict = {
            "attention":{},
            "spatial":{},
            "contacting":{},
        }
        self.rel_num = {'attention': 3,
        'spatial': 6,
        'contacting': 17}
        for rel in self.rel_num.keys():
            for i in range(self.rel_num[rel]):
                self.rel_num_dict[rel][i] =1
    def update_dict(self,targets,rel):
        batch_size = targets.size(0)
        for i in range(batch_size):
            gt_i = int(targets[i].argmax())
            self.rel_num_dict[rel][gt_i]+=1
        return
    def compute_alpha(self,rel,beta =0.5):
        class_num    = self.rel_num[rel]
        alpha = torch.zeros(1,class_num)
        for cls,num in self.rel_num_dict[rel].items():
            alpha[0,cls] = num
        alpha = (1-beta)/(1-beta**alpha)
        return alpha

    def forward(self, logits, targets, rel):
        # 更新数量字典
        self.update_dict(targets,rel)
        # 计算alpha
        alpha = self.compute_alpha(rel)
        self.alpha = alpha.to(logits.device)
        sigmoid_p = torch.sigmoid(logits)
        p_t = (targets * sigmoid_p) + ((1 - targets) * (1 - sigmoid_p))
        

        modulating_factor = (torch.pow(1 - p_t, self.gamma_pos)*targets)+(torch.pow(1 - p_t, self.gamma_pos)*(1-targets))
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if self.alpha is not None:
            alpha_t = self.alpha * targets 
            focal_loss = modulating_factor * alpha_t * ce_loss
        else:
            focal_loss = modulating_factor * ce_loss

        if self.reduction =='mean':
            return focal_loss.mean()
        elif self.reduction =='sum':
            return focal_loss.sum()
        else:
            return focal_loss