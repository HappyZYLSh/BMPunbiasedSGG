"""
Let's get the relationships yo
"""

import math
from re import U
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import math

from lib.word_vectors import obj_edge_vectors
from lib.transformer import transformer
from lib.fpn.box_utils import center_size
from lib.gmm_heads import *
from fasterRCNN.lib.model.roi_layers import ROIAlign, nms
from lib.draw_rectangles.draw_rectangles import draw_union_boxes

EncoderLayer = nn.TransformerEncoderLayer
Encoder = nn.TransformerEncoder
class PrototypeVAE(nn.Module):
    def __init__(self, rel=None, input_dim=1936, hidden_dim=256, latent_dim=40, prototype_dim=1936, alpha=0.05, tau_maj_ratio=0.5, q=3,n_stage = 6):
        # prototype_dim: dimension of prototype features
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim   
        self.hidden_dim1 = int(input_dim * 1.5) 
        self.rel = rel
        self.num_classes = {'attention': 3,
                           'spatial': 6,
                           'contacting': 17}
        self.num_class = self.num_classes[self.rel]
        self.class_counts = {
            'attention': [55884, 77705, 12970],
            'spatial': [2307, 24680, 97125, 18881, 28123, 4778],
            'contacting': [1797, 1899, 1450, 1106, 161, 59470, 4057, 1338, 40733, 3724, 16614, 3379, 19667, 27, 2358, 255, 277]
        }
        self.class_count = self.class_counts[self.rel]
        self.nstage = n_stage
        
        # Prototype converter: from probability prototypes to Gaussian prototype parameters
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, input_dim*2)  # mu 
        )
        
        # Decoder: from Gaussian prototypes to generated samples
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # reconstructed input
        )

        # Learnable class prototype storage (base)
        self.prototypes_base = []
        self.prototypes_gaussian = {
            "attention": None,
            "spatial": None,
            "contacting": None
        }
        self.alpha = alpha  # weight for prototype update
        self.q = q  # number of nearest majority prototypes to consider
        self.T_maj = tau_maj_ratio  # threshold ratio for majority class

        self.flag = []

    def reparameterize(self, mu, log_sigma):
        epsilon = torch.randn_like(log_sigma)
        return mu + torch.exp(0.5 * log_sigma) * epsilon

    def get_gaussian_prototypes(self):
        """
        Convert probability prototypes to Gaussian prototypes
        """
        prototypes = self.encoder(self.prototypes_base[self.rel])
        return prototypes

    def compute_KL_distance_matrix(self, P1, P2):
        """
        Compute the n1×n2 Kullback–Leibler (KL) Divergence  matrix between two tensors P1 (n1, 2*d) and P2 (n2, 2*d).
        """
        mu1 = P1[:,:self.input_dim]
        sigma1_sq = torch.exp(0.5 * P1[:,self.input_dim:])

        mu2 = P2[:,:self.input_dim]
        sigma2_sq = torch.exp(0.5 * P2[:,self.input_dim:])

        n1, d = mu1.shape
        n2, d2 = mu2.shape

        assert d == d2, "特征维度 d 必须相同"
        assert sigma1_sq.shape == mu1.shape, "sigma1_sq 必须与 mu1 同形状"
        assert sigma2_sq.shape == mu2.shape, "sigma2_sq 必须与 mu2 同形状"

        # 利用广播机制计算所有 (i,j) 对
        # mu1:    [n1, d]
        # mu2:    [n2, d] --> 扩展为 [1, n2, d]
        # sigma1_sq: [n1, d] --> 扩展为 [n1, 1, d]
        # sigma2_sq: [n2, d] --> 扩展为 [1, n2, d]

        mu1_exp = mu1.unsqueeze(1)      # shape: (n1, 1, d)
        mu2_exp = mu2.unsqueeze(0)      # shape: (1, n2, d)

        sigma1_sq_exp = sigma1_sq.unsqueeze(1)  # shape: (n1, 1, d)
        sigma2_sq_exp = sigma2_sq.unsqueeze(0)  # shape: (1, n2, d)

        # KL 散度的三个部分
        term1 = torch.log(sigma2_sq_exp / (sigma1_sq_exp + 1e-12))  # log(σ2² / σ1²)
        term2 = sigma1_sq_exp / (sigma2_sq_exp + 1e-12)             # σ1² / σ2²
        term3 = (mu2_exp - mu1_exp) **2 / (sigma2_sq_exp + 1e-12)  # (μ2 - μ1)² / σ2²

        # 按维度求和：对 d 求和
        sum_term1 = torch.sum(term1, dim=2)  # shape: (n1, n2)
        sum_term2 = torch.sum(term2, dim=2)
        sum_term3 = torch.sum(term3, dim=2)

        kl_matrix = 0.5 * (sum_term1 + sum_term2 + sum_term3 - d)  # shape: (n1, n2)
        return kl_matrix

    def update_minority_prototypes(self, mu_log_sigma, Cmin=None, Cmaj=None):
        """
        Update minority class prototypes based on nearest majority class prototypes
        """
        alpha = self.alpha
        q = self.q

        prototype_matrix = self.compute_KL_distance_matrix(
            mu_log_sigma[Cmin, :],
            mu_log_sigma[Cmaj, :]
        )  # Shape: [n_minor, n_major]

        # Find the nearest q majority prototypes for each minority prototype
        Set_distance, Set_c = torch.topk(prototype_matrix, k=min(len(Cmaj), q) if len(Cmaj) < q else q, largest=False, sorted=False)

        for i, cls in enumerate(Cmin):
            x = mu_log_sigma[cls].unsqueeze(0)  # Shape: [1, d]
            nearest_indices = [Cmaj[idx] for idx in Set_c[i, :]]
            x_nearest = mu_log_sigma[nearest_indices]  # Shape: [q, d]

            x = (1 - alpha) * x
            nearest_Maj_index = [Cmaj[idx] for idx in Set_c[i, :]]
            nearest_num = torch.tensor([self.class_count[idx] for idx in nearest_Maj_index], device=Set_c.device).unsqueeze(1)  # Shape: [q, 1]
            nearest_distance = Set_distance[i].unsqueeze(1)  # Shape: [q, 1]  

            # Assuming you want to use distances from the current cls to nearest majors
            # Corrected: use the corresponding distances for this cls
            current_distances = prototype_matrix[i, :len(nearest_indices)]  # Shape: [q]
            current_distances = current_distances.unsqueeze(1)  # Shape: [q, 1]
            nearest_num = torch.tensor([self.class_count[idx] for idx in nearest_Maj_index], device=current_distances.device).unsqueeze(1)  # Shape: [q, 1]

            param1 = current_distances * nearest_num  # Shape: [q, 1]
            weight = param1 / torch.sum(param1, dim=0, keepdim=True)  # Shape: [q, 1]
            param2 = torch.sum(weight * x_nearest, dim=0).unsqueeze(0)  # Shape: [1, d]
            param2 = alpha * param2
            new_mu_sigma = x + param2  # Shape: [1, d]
            mu_log_sigma[cls] = new_mu_sigma

        self.prototypes_gaussian[self.rel] = mu_log_sigma  # Shape: [num_classes, input_dim*2]
        return mu_log_sigma

    def loss_function(self, mu, logvar, recon_x, recon_y):
        x_recon = recon_x  # Shape: [B*stage, d]
        x = self.prototypes_base[self.rel][recon_y]  # Shape: [B*stage, d]
        recon_loss = nn.BCEWithLogitsLoss(reduction='mean')  # default is mean

        kl_loss = torch.mean(0.5 * (mu.pow(2) + logvar.exp() - logvar - 1))
    
        return recon_loss(x_recon, x) + 0.5 * kl_loss

    def forward(self, class_num):
        # mus = []
        # log_sigmas = []
        # recon_xs = []
        # recon_ys = []
        # Step 1: Identify majority and minority classes
        tau_maj_ratio = self.T_maj

        Cmaj = []  # List of majority classes
        Cmin = []  # List of minority classes
        for i, num in enumerate(self.class_count):
            # if num >= tau_maj_ratio * max(self.class_count):
            if num >= 20000:

                Cmaj.append(i)
            else:
                Cmin.append(i)
        Flag = torch.zeros(1, self.num_class)
        for mini in Cmin:
            Flag[0, mini] = 1
            

        for nn in range(self.nstage):
            if nn==0:

                # Step 2: Initialize Gaussian prototypes
                prototypes = self.get_gaussian_prototypes()  # Shape: [num_classes, input_dim*2]
                p = prototypes.detach().clone()

            # Step 3: Update minority Gaussian prototypes and store in self.prototypes_gaussian
            if Cmaj and Cmin:
                prototypes = self.update_minority_prototypes(mu_log_sigma=prototypes, Cmaj=Cmaj, Cmin=Cmin)  # Shape: [num_classes, input_dim*2]

            # Step 4: Compute sampling weights for minority classes and generate class indices
            k_sample = [(20000-self.class_count[i]) / self.class_count[i] for i in Cmin]
            class_ids = []
            for k, i in zip(k_sample, Cmin):
                k_num = int(k * class_num[i])
                class_ids += [i for _ in range(k_num)]  # Shape: [batch_size = (20000-self.class_count[i]) / self.class_count[i]*class_num[i],]

            # Step 5: Sample from prototypes based on class_ids
            mu = prototypes[class_ids, :self.input_dim]  # Shape: [batch_size, input_dim]
            log_sigma = prototypes[class_ids, self.input_dim:]  # Shape: [batch_size, input_dim]

        z = self.reparameterize(mu, log_sigma)  # Shape: [batch_size, input_dim]

        # Step 6: Generate synthetic samples
        decoder_input = z
        recon_x = self.decoder(decoder_input)  # Shape: [batch_size, input_dim]
        recon_y = class_ids

        # Ensure the shapes are consistent
        assert recon_x.shape[0] == len(recon_y), f"recon_x.shape[0]: {recon_x.shape[0]}, len(recon_y): {len(recon_y)}"

        # Step 7: Compute reconstruction loss
        loss = self.loss_function(mu, log_sigma, recon_x, recon_y)
        return recon_x, mu, log_sigma, recon_y, Flag, loss
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, indices=None) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if indices is None:
            x = x + self.pe[:,:x.size(1)]
        else:
            pos = torch.cat([self.pe[:, index] for index in indices])            
            x = x + pos
        return self.dropout(x)

class ObjectClassifier(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, mode='sgdet', obj_head='gmm', K=4, obj_classes=None, 
                mem_compute=None, selection=None, selection_lambda=0.5,
                tracking=None):
        super(ObjectClassifier, self).__init__()
        self.classes = obj_classes
        self.mode = mode
        self.GMM_K =K
        self.obj_memory = []
        self.mem_compute = mem_compute
        self.selection = selection
        #----------add nms when sgdet
        self.nms_filter_duplicates = True
        self.max_per_img =64
        self.thresh = 0.01

        #roi align
        self.RCNN_roi_align = ROIAlign((7, 7), 1.0/16.0, 0)

        embed_vecs = obj_edge_vectors(obj_classes[1:], wv_type='glove.6B', wv_dir='data', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes)-1, 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(nn.BatchNorm1d(4, momentum=0.01 / 10.0),
                                       nn.Linear(4, 128),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1))
        self.obj_dim = 2048
        self.obj_head = obj_head
        self.tracking = tracking
        mem_embed = 1024
        if self.tracking:
            d_model = self.obj_dim + 200 + 128
            encoder_layer = EncoderLayer(d_model=d_model, dim_feedforward=1024, nhead=8, batch_first=True)
            self.positional_encoder = PositionalEncoding(d_model, 0.1, 600 if mode=="sgdet" else 400)
            self.encoder_tran = Encoder(encoder_layer, num_layers=3)
            mem_embed = d_model
        if mem_compute:
                self.mem_attention = nn.MultiheadAttention(mem_embed, 1, 0.0, bias=False)
               
                if selection == 'manual':
                    self.selector = selection_lambda
                else:
                    self.selector = nn.Linear(1024,1)
        if obj_head == 'gmm':
            self.intermediate =  nn.Sequential(nn.Linear(self.obj_dim + 200 + 128, 1024),
                                               nn.BatchNorm1d(1024),
                                               nn.ReLU())
            self.decoder_lin = GMM_head(hid_dim=1024, num_classes=len(self.classes), rel_type=None, k=self.GMM_K)

        else:
            self.intermediate =  nn.Sequential(nn.Linear(self.obj_dim + 200 + 128, 1024),
                                               nn.BatchNorm1d(1024),
                                               nn.ReLU())
            self.decoder_lin = nn.Sequential(nn.Linear(1024, len(self.classes)))

    def clean_class(self, entry, b, class_idx):
        final_boxes = []
        final_dists = []
        final_feats = []
        final_mem_feats = []
        final_labels = []
        for i in range(b):
            scores = entry['distribution'][entry['boxes'][:, 0] == i]
            pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i]
            feats = entry['features'][entry['boxes'][:, 0] == i]

            if 'object_mem_features' in entry.keys():
                mem_feats = entry['object_mem_features'][entry['boxes'][:, 0] == i]
            else:
                mem_feats = feats
            pred_labels = entry['pred_labels'][entry['boxes'][:, 0] == i]

            new_box = pred_boxes[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_feats = feats[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_mem_feats = mem_feats[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_scores = scores[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_scores[:, class_idx-1] = 0
            if new_scores.shape[0] > 0:
                new_labels = torch.argmax(new_scores, dim=1) + 1
            else:
                new_labels = torch.tensor([], dtype=torch.long).cuda(0)

            final_dists.append(scores)
            final_dists.append(new_scores)
            final_boxes.append(pred_boxes)
            final_boxes.append(new_box)
            final_feats.append(feats)
            final_feats.append(new_feats)
            final_mem_feats.append(mem_feats)
            final_mem_feats.append(new_mem_feats)
            final_labels.append(pred_labels)
            final_labels.append(new_labels)

        entry['boxes'] = torch.cat(final_boxes, dim=0)
        entry['distribution'] = torch.cat(final_dists, dim=0)
        entry['features'] = torch.cat(final_feats, dim=0)
        entry['object_mem_features'] = torch.cat(final_mem_feats, dim=0)
        entry['pred_labels'] = torch.cat(final_labels, dim=0)
        return entry

    def mem_selection(self,feat):
        if self.selection == 'manual':
            return self.selector
        else:
            return self.selector(feat).sigmoid()

    def memory_hallucinator(self,memory,feat):
        if len(memory) != 0:
            e = self.mem_selection(feat)
            q = feat.unsqueeze(1)
            
            k = v = memory.unsqueeze(1)
            mem_features,_ = self.mem_attention(q,k,v)

            if e is not None:
                mem_encoded_features = e*feat + (1-e)*mem_features.squeeze(1)
            else:
                mem_encoded_features = feat + mem_features.squeeze(1)
            # mem_encoded_features = feat + e*mem_features.squeeze(1)
        else:
            mem_encoded_features = feat

        return mem_encoded_features
    def classify(self,entry,obj_features,phase='train',unc=False):
        if self.tracking:
            indices = entry["indices"]

                # save memory by filetering out single-element sequences, indices[0]
            final_features = torch.zeros_like(obj_features).to(obj_features.device)
            if len(indices)>1:
                pos_index = []
                for index in indices[1:]:
                    im_idx, counts = torch.unique(entry["boxes"][index][:,0].view(-1), return_counts=True, sorted=True)
                    counts = counts.tolist()
                    pos = torch.cat([torch.LongTensor([im]*count) for im, count in zip(range(len(counts)), counts)])
                    pos_index.append(pos)
                sequence_features = pad_sequence([obj_features[index] for index in indices[1:]], batch_first=True)
                masks = (1-pad_sequence([torch.ones(len(index)) for index in indices[1:]], batch_first=True)).bool()
                pos_index = pad_sequence(pos_index, batch_first=True)
                obj_ = self.encoder_tran(self.positional_encoder(sequence_features, pos_index),src_key_padding_mask=masks.cuda())
                obj_flat = torch.cat([obj[:len(index)]for index, obj in zip(indices[1:],obj_)])
                indices_flat = torch.cat(indices[1:]).unsqueeze(1).repeat(1,obj_features.shape[1])
                final_features.scatter_(0, indices_flat, obj_flat)
            if len(indices[0]) > 0:
                non_ = self.encoder_tran(self.positional_encoder(obj_features[indices[0]].unsqueeze(1)))           
                final_features.scatter_(0, indices[0].unsqueeze(1).repeat(1,obj_features.shape[1]), non_[:,0,:])

            obj_features = final_features
            
            entry['object_features'] = obj_features
            if self.mem_compute:
                obj_features = self.memory_hallucinator(memory=self.obj_memory, feat=obj_features)
            entry['object_mem_features'] = obj_features
            obj_features = self.intermediate(obj_features)
            
        else:
            obj_features = self.intermediate(obj_features)
            entry['object_features'] = obj_features
            if self.mem_compute:
                obj_features = self.memory_hallucinator(memory=self.obj_memory, feat=obj_features)
            entry['object_mem_features'] = obj_features

        if phase == 'train':           
            if self.obj_head == 'gmm':
                if not unc:
                    entry['distribution'] = self.decoder_lin(obj_features,phase=phase,unc=unc)
                else:
                    entry['distribution'] = self.decoder_lin(obj_features,phase='test',unc=False)
                    entry['obj_al_uc'],entry['obj_ep_uc'] = self.decoder_lin(obj_features,unc=unc)
            else:
                entry['distribution'] = self.decoder_lin(obj_features)
            entry['pred_labels'] = entry['labels']
        else:
            if self.obj_head == 'gmm':
                entry['distribution'] = self.decoder_lin(obj_features,phase=phase,unc=unc)
            else:
                entry['distribution'] = self.decoder_lin(obj_features)
                entry['distribution'] = torch.softmax(entry['distribution'][:, 1:],dim=1)
        return entry

    def forward(self, entry, phase='train', unc=False):

        if self.mode  == 'predcls':
            entry['pred_labels'] = entry['labels']
            return entry
        elif self.mode == 'sgcls':
            obj_embed = entry['distribution'] @ self.obj_embed.weight
            pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
            obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1)
            if phase == 'train':
                # obj_features = self.intermediate(obj_features)
                # entry['object_features'] = obj_features
                # if self.mem_compute:
                #     obj_features = self.memory_hallucinator(memory=self.obj_memory, feat=obj_features)
                # entry['object_mem_features'] = obj_features
                # if self.obj_head == 'gmm':
                #     if not unc:
                #         entry['distribution'] = self.decoder_lin(obj_features,phase=phase,unc=unc)
                #     else:
                #         entry['distribution'] = self.decoder_lin(obj_features,phase='test',unc=False)
                #         entry['obj_al_uc'],entry['obj_ep_uc'] = self.decoder_lin(obj_features,unc=unc)
                # else:
                #     entry['distribution'] = self.decoder_lin(obj_features)
                # entry['pred_labels'] = entry['labels']
                entry = self.classify(entry,obj_features,phase,unc)
            else:
                # obj_features = self.intermediate(obj_features)
                # if self.mem_compute:
                #     obj_features = self.memory_hallucinator(memory=self.obj_memory, feat=obj_features)
                # entry['object_mem_features'] = obj_features
                # if self.obj_head == 'gmm':
                #     entry['distribution'] = self.decoder_lin(obj_features,phase=phase,unc=unc)
                # else:
                #     entry['distribution'] = self.decoder_lin(obj_features)
                #     entry['distribution'] = torch.softmax(entry['distribution'][:, 1:],dim=1)
                entry = self.classify(entry,obj_features,phase,unc)

                box_idx = entry['boxes'][:,0].long()
                
                b = int(box_idx[-1] + 1)

                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2

                # use the infered object labels for new pair idx
                HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(obj_features.device)
                global_idx = torch.arange(0, entry['boxes'].shape[0])

                for i in range(b):
                    local_human_idx = torch.argmax(entry['distribution'][box_idx == i, 0]) # the local bbox index with highest human score in this frame
                    HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

                entry['pred_labels'][HUMAN_IDX.squeeze()] = 1
                entry['pred_scores'][HUMAN_IDX.squeeze()] = entry['distribution'][HUMAN_IDX.squeeze(), 0]

                # drop repeat overlap TODO!!!!!!!!!!!!
                for i in range(b):
                    duplicate_class = torch.mode(entry['pred_labels'][entry['boxes'][:, 0] == i])[0]
                    present = entry['boxes'][:, 0] == i
                    if torch.sum(entry['pred_labels'][entry['boxes'][:, 0] == i] ==duplicate_class) > 0:
                        duplicate_position = entry['pred_labels'][present] == duplicate_class

                        ppp = torch.argsort(entry['distribution'][present][duplicate_position][:,duplicate_class - 1])[:-1]
                        for j in ppp:

                            changed_idx = global_idx[present][duplicate_position][j]
                            entry['distribution'][changed_idx, duplicate_class-1] = 0
                            entry['pred_labels'][changed_idx] = torch.argmax(entry['distribution'][changed_idx])+1
                            entry['pred_scores'][changed_idx] = torch.max(entry['distribution'][changed_idx])


                im_idx = []  # which frame are the relations belong to
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    for m in global_idx[box_idx==j][entry['pred_labels'][box_idx==j] != 1]: # this long term contains the objects in the frame
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                pair = torch.tensor(pair).to(obj_features.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(obj_features.device)
                entry['pair_idx'] = pair
                entry['im_idx'] = im_idx

                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] * entry['im_info']
                union_boxes = torch.cat((im_idx[:, None], torch.min(entry['boxes'][:, 1:3][pair[:, 0]], entry['boxes'][:, 1:3][pair[:, 1]]),
                                        torch.max(entry['boxes'][:, 3:5][pair[:, 0]], entry['boxes'][:, 3:5][pair[:, 1]])), 1)

                union_feat = self.RCNN_roi_align(entry['fmaps'], union_boxes)
                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] / entry['im_info']
                pair_rois = torch.cat((entry['boxes'][pair[:, 0], 1:], entry['boxes'][pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(obj_features.device)
                entry['union_feat'] = union_feat
                entry['union_box'] = union_boxes
                entry['spatial_masks'] = spatial_masks
            return entry
        else:
            obj_embed = entry['distribution'] @ self.obj_embed.weight
            pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
            obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1)
            if phase == 'train':
                entry = self.classify(entry,obj_features,phase,unc)

                # box_idx = entry['boxes'][:, 0][entry['pair_idx'].unique()]
                # l = torch.sum(box_idx == torch.mode(box_idx)[0])
                # b = int(box_idx[-1] + 1)  # !!!

                # obj_features = self.intermediate(obj_features)
                # entry['object_features'] = obj_features
                # if self.mem_compute:
                #     obj_features = self.memory_hallucinator(memory=self.obj_memory, feat=obj_features)
                # entry['object_mem_features'] = obj_features
                # if self.obj_head == 'gmm':
                #     if not unc:
                #         entry['distribution'] = self.decoder_lin(obj_features,phase=phase,unc=unc)
                #     else:
                #         entry['distribution'] = self.decoder_lin(obj_features,phase='test',unc=False)
                #         entry['obj_al_uc'],entry['obj_ep_uc'] = self.decoder_lin(obj_features,unc=unc)
                # else:
                #     entry['distribution'] = self.decoder_lin(obj_features)
                # entry['pred_labels'] = entry['labels']
            else:
                entry = self.classify(entry,obj_features,phase,unc)
                # if self.mem_compute:
                #     obj_features = self.memory_hallucinator(memory=self.obj_memory, feat=obj_features)
                # entry['object_mem_features'] = obj_features
                # if self.obj_head == 'gmm':
                #     obj_features = self.intermediate(obj_features)
                #     entry['distribution'] = self.decoder_lin(obj_features,phase=phase,unc=unc)[:, 1:]
                # else:
                #     obj_features = self.intermediate(obj_features)
                #     entry['distribution'] = self.decoder_lin(obj_features)
                #     entry['distribution'] = torch.softmax(entry['distribution'][:, 1:], dim=1)

                box_idx = entry['boxes'][:, 0].long()
                b = int(box_idx[-1] + 1)

                entry = self.clean_class(entry, b, 5)
                entry = self.clean_class(entry, b, 8)
                entry = self.clean_class(entry, b, 17)

                # # NMS
                final_boxes = []
                final_dists = []
                final_feats = []
                final_mem_feats = []
                for i in range(b):
                    # images in the batch
                    scores = entry['distribution'][entry['boxes'][:, 0] == i]
                    pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i, 1:]
                    feats = entry['features'][entry['boxes'][:, 0] == i]
                    if 'object_mem_features' in entry.keys():
                        mem_feats = entry['object_mem_features'][entry['boxes'][:, 0] == i]
                    else:
                        mem_feats = entry['features'][entry['boxes'][:, 0] == i]
                    for j in range(len(self.classes) - 1):
                        # print('scores_shape: ',scores.shape,' for class ',j)
                        # NMS according to obj categories
                        if scores.numel() == 0:
                            inds = torch.empty(scores.shape)
                        else:
                            inds = torch.nonzero(torch.argmax(scores, dim=1) == j).view(-1)
                        # if there is det
                        if inds.numel() > 0:
                            cls_dists = scores[inds]
                            cls_feats = feats[inds]
                            cls_mem_feats = mem_feats[inds]
                            cls_scores = cls_dists[:, j]
                            _, order = torch.sort(cls_scores, 0, True)
                            cls_boxes = pred_boxes[inds]
                            cls_dists = cls_dists[order]
                            cls_feats = cls_feats[order]
                            cls_mem_feats = cls_mem_feats[order]
                            keep = nms(cls_boxes[order, :], cls_scores[order], 0.6)  # hyperparameter

                            final_dists.append(cls_dists[keep.view(-1).long()])
                            final_boxes.append(torch.cat((torch.tensor([[i]],dtype=torch.float).repeat(keep.shape[0],1).cuda(0), cls_boxes[order, :][keep.view(-1).long()]), 1))
                            final_feats.append(cls_feats[keep.view(-1).long()])
                            final_mem_feats.append(cls_mem_feats[keep.view(-1).long()])

                entry['boxes'] = torch.cat(final_boxes, dim=0)
                box_idx = entry['boxes'][:, 0].long()
                entry['distribution'] = torch.cat(final_dists, dim=0)
                entry['features'] = torch.cat(final_feats, dim=0)
                entry['object_mem_features'] = torch.cat(final_mem_feats, dim=0)

                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2

                # use the infered object labels for new pair idx
                HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(box_idx.device)
                global_idx = torch.arange(0, entry['boxes'].shape[0])

                for i in range(b):
                    if entry['distribution'][ box_idx == i, 0].numel() > 0:
                        local_human_idx = torch.argmax(entry['distribution'][box_idx == i, 0])  # the local bbox index with highest human score in this frame
                        HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

                entry['pred_labels'][HUMAN_IDX.squeeze()] = 1
                entry['pred_scores'][HUMAN_IDX.squeeze()] = entry['distribution'][HUMAN_IDX.squeeze(), 0]

                im_idx = []  # which frame are the relations belong to
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    for m in global_idx[box_idx == j][
                        entry['pred_labels'][box_idx == j] != 1]:  # this long term contains the objects in the frame
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                pair = torch.tensor(pair).to(box_idx.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(box_idx.device)
                entry['pair_idx'] = pair
                entry['im_idx'] = im_idx
                entry['human_idx'] = HUMAN_IDX
                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] * entry['im_info']
                union_boxes = torch.cat(
                    (im_idx[:, None], torch.min(entry['boxes'][:, 1:3][pair[:, 0]], entry['boxes'][:, 1:3][pair[:, 1]]),
                     torch.max(entry['boxes'][:, 3:5][pair[:, 0]], entry['boxes'][:, 3:5][pair[:, 1]])), 1)

                union_feat = self.RCNN_roi_align(entry['fmaps'], union_boxes)
                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] / entry['im_info']
                entry['union_feat'] = union_feat
                entry['union_box'] = union_boxes
                pair_rois = torch.cat((entry['boxes'][pair[:, 0], 1:], entry['boxes'][pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                entry['spatial_masks'] = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(box_idx.device)

            return entry


class BMP(nn.Module):

    def __init__(self, mode='sgdet',attention_class_num=None, spatial_class_num=None, \
                 contact_class_num=None, obj_classes=None,
                 rel_classes=None,enc_layer_num=None, dec_layer_num=None, obj_mem_compute=None,rel_mem_compute=None,
                 mem_fusion=None,selection=None,selection_lambda=0.5,take_obj_mem_feat=False,
                 obj_head = 'gmm', rel_head = 'gmm',K =None, tracking=None,
                 nstage = 6,q = 3,T_maj = [0.5,0.25,0.25],alpha = 0.7):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        """
        super(BMP, self).__init__()
        self.obj_classes = obj_classes
        self.GMM_K = K
        self.mem_fusion = mem_fusion
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        assert mode in ('sgdet', 'sgcls', 'predcls')
        self.mode = mode
        self.tracking = tracking
        self.take_obj_mem_feat = take_obj_mem_feat

        self.obj_head = obj_head
        self.rel_head = rel_head
        self.obj_mem_compute = obj_mem_compute
        self.rel_mem_compute = rel_mem_compute

        self.selection_lambda = selection_lambda
        self.rel_memory = []
        self.object_classifier = ObjectClassifier(mode=self.mode, obj_classes=self.obj_classes, 
                                                obj_head = obj_head, mem_compute=obj_mem_compute, K=K, selection=selection,
                                                selection_lambda=selection_lambda, tracking=self.tracking)

        ###################################
        self.union_func1 = nn.Conv2d(1024, 256, 1, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 256 //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
        )
        if not take_obj_mem_feat:
            self.subj_fc = nn.Linear(2048, 512)
            self.obj_fc = nn.Linear(2048, 512)
        else:
            if self.tracking:
                self.subj_fc = nn.Linear(2048+200+128, 512)
                self.obj_fc = nn.Linear(2048+200+128, 512)
            else:
                self.subj_fc = nn.Linear(1024, 512)
                self.obj_fc = nn.Linear(1024, 512)

        self.vr_fc = nn.Linear(256*7*7, 512)

        embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='data/', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
        self.obj_embed2.weight.data = embed_vecs.clone()

            
        self.glocal_transformer = transformer(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num,
                                              embed_dim=1936, nhead=8,
                                              dim_feedforward=2048, dropout=0.1, mode='latter', 
                                              mem_compute=rel_mem_compute, mem_fusion=mem_fusion,
                                              selection=selection, selection_lambda=self.selection_lambda,
                                              )

        if rel_head == 'gmm':
            self.a_rel_compress = GMM_head(hid_dim=1936, num_classes=self.attention_class_num, rel_type='attention', k=self.GMM_K)
            self.s_rel_compress = GMM_head(hid_dim=1936, num_classes=self.spatial_class_num, rel_type='spatial', k=self.GMM_K)
            self.c_rel_compress = GMM_head(hid_dim=1936, num_classes=self.contact_class_num, rel_type='contact', k=self.GMM_K)

        else:
            self.a_rel_compress = nn.Linear(1936, self.attention_class_num)
            self.s_rel_compress = nn.Linear(1936, self.spatial_class_num)
            self.c_rel_compress = nn.Linear(1936, self.contact_class_num)
        
        # PrototypeVAE
        self.nstage = nstage
        self.a_PrototypeVAE = PrototypeVAE(rel="attention",alpha=alpha,tau_maj_ratio=T_maj[0],q=q,nstage = nstage)
        self.s_PrototypeVAE = PrototypeVAE(rel="spatial",alpha=alpha,tau_maj_ratio=T_maj[1],q=q,nstage = nstage)
        self.c_PrototypeVAE = PrototypeVAE(rel="contacting",alpha=alpha,tau_maj_ratio=T_maj[2],q=q,nstage = nstage)

    def forward(self, entry, phase='train',unc=False):

        entry = self.object_classifier(entry, phase=phase, unc=unc)
        # visual part
        if not self.take_obj_mem_feat:
            subj_rep = entry['features'][entry['pair_idx'][:, 0]]
            obj_rep = entry['features'][entry['pair_idx'][:, 1]]
        else:
            subj_rep = entry['object_mem_features'][entry['pair_idx'][:, 0]]
            obj_rep = entry['object_mem_features'][entry['pair_idx'][:, 1]]

        subj_rep = self.subj_fc(subj_rep)
        obj_rep = self.obj_fc(obj_rep)
        vr = self.union_func1(entry['union_feat']) + self.conv(entry['spatial_masks'])
        vr = self.vr_fc(vr.view(-1,256*7*7))
        x_visual = torch.cat((subj_rep, obj_rep, vr), 1)

        # semantic part
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]

        subj_emb = self.obj_embed(subj_class)
        obj_emb = self.obj_embed2(obj_class)
        x_semantic = torch.cat((subj_emb, obj_emb), 1)

        rel_features = torch.cat((x_visual, x_semantic), dim=1)

        # Spatial-Temporal Transformer
        global_output,rel_features,mem_features, _, _ = \
        self.glocal_transformer(features=rel_features, im_idx=entry['im_idx'], memory=self.rel_memory)

        
        entry["rel_features"] = rel_features
        entry['rel_mem_features'] = mem_features
        if self.a_PrototypeVAE.prototypes_base and self.s_PrototypeVAE.prototypes_base and self.c_PrototypeVAE.prototypes_base:
            attention_label = torch.tensor(entry["attention_gt"], dtype=torch.long).squeeze()
            # bce loss
            spatial_label = torch.zeros([len(entry["spatial_gt"]), 6], dtype=torch.float32)
            contact_label = torch.zeros([len(entry["contacting_gt"]), 17], dtype=torch.float32)
            for i in range(len(entry["spatial_gt"])):
                spatial_label[i, entry["spatial_gt"][i]] = 1
                contact_label[i, entry["contacting_gt"][i]] = 1

            class_num = {
                "attention":[],
                "spatial":[],
                "contacting":[],
            }

            for rel,label,class_count in zip(["attention","spatial","contacting"],[attention_label,spatial_label,contact_label],[3,6,17]):
                for i in range(class_count):
                    if rel=="attention":
                        class_num[rel] = class_num[rel]+[torch.sum(torch.eq(label,i)).item()]
                    else:
                        class_num[rel] = class_num[rel]+[torch.sum(label[:,i]).item()]
            
            recon_x1, mu1, log_sigma1,recon_y1,Flag1,loss1   = self.a_PrototypeVAE(class_num = class_num["attention"])
            
            recon_x2, mu2, log_sigma2,recon_y2,Flag2,loss2 = self.s_PrototypeVAE(class_num = class_num["spatial"])

            recon_x3, mu3, log_sigma3,recon_y3,Flag3,loss3 = self.c_PrototypeVAE(class_num = class_num["contacting"])
            entry["attention_maj"] = 1-Flag1
            entry["spatial_maj"] = 1-Flag2
            entry["contacting_maj"] = 1-Flag3


            if len(recon_y1)>0:
                # x1 = 
                entry["recon_attention_f"] = recon_x1
                entry["recon_attention_gt"] = recon_y1
                entry["recon_attention_log_sigma"] = log_sigma1
                entry["recon_attention_mu"] = mu1  
                entry["flag1"] = True
                entry["loss1"] = loss1
            else:
                entry["flag1"] = False


            if len(recon_y2)>0:
                entry["recon_spatial_f"] = recon_x2
                entry["recon_spatial_mu"] = mu2
                entry["recon_spatial_gt"] = [ [y] for y in recon_y2]
                entry["recon_spatial_log_sigma"] = log_sigma2
                entry["flag2"] = True
                entry["loss2"] = loss2
            else:
                entry["flag2"] = False

            if len(recon_y3)>0:
                entry["recon_contacting_f"] = recon_x3
                entry["recon_contacting_mu"] = mu3
                entry["recon_contacting_log_sigma"] = log_sigma3
                entry["recon_contacting_gt"] = [ [y] for y in recon_y3]
                entry["flag3"] = True
                entry["loss3"] = loss3
            else:
                entry["flag3"] = False


        if self.rel_head == 'gmm':
            if not unc:
                entry["attention_distribution"] = self.a_rel_compress(global_output,phase,unc)
                entry["spatial_distribution"] = self.s_rel_compress(global_output,phase,unc)
                entry["contacting_distribution"] = self.c_rel_compress(global_output,phase,unc)
                if self.a_PrototypeVAE.prototypes_base and self.s_PrototypeVAE.prototypes_base and self.c_PrototypeVAE.prototypes_base:
                    if len(recon_y1)>0:

                        entry["recon_attention_distribution"] = self.a_rel_compress(recon_x1,phase,unc)
                    if len(recon_y2)>0:

                        entry["recon_spatial_distribution"] = self.s_rel_compress(recon_x2,phase,unc)
                    if len(recon_y3)>0:

                        entry["recon_contacting_distribution"] = self.c_rel_compress(recon_x3,phase,unc)
            else:
                entry["attention_al_uc"], entry["attention_ep_uc"] = self.a_rel_compress(global_output,phase,unc)
                entry["spatial_al_uc"], entry["spatial_ep_uc"] = self.s_rel_compress(global_output,phase,unc)
                entry["contacting_al_uc"], entry["contacting_ep_uc"] = self.c_rel_compress(global_output,phase,unc)
        else:
            entry["attention_distribution"] = self.a_rel_compress(global_output)
            if phase == 'test':
                entry["attention_distribution"] = entry["attention_distribution"].softmax(-1)
            entry["spatial_distribution"] = self.s_rel_compress(global_output)
            entry["contacting_distribution"] = self.c_rel_compress(global_output)
            entry["spatial_distribution"] = torch.sigmoid(entry["spatial_distribution"])
            entry["contacting_distribution"] = torch.sigmoid(entry["contacting_distribution"])

        return entry

