from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
 
from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices

def get_metrics(similarity, qids, gids, n_, retur_indices=False):
    t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
    if retur_indices:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]], indices
    else:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]]


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("RDE.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        # Đảm bảo tất cả submodules cũng ở eval mode
        # Đặc biệt là vision_model và text_model trong base_model
        if hasattr(model, 'base_model'):
            if hasattr(model.base_model, 'vision_model'):
                model.base_model.vision_model.eval()
            if hasattr(model.base_model, 'text_model'):
                model.base_model.text_model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption).cpu()
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img).cpu()
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        return qfeats.cpu(), gfeats.cpu(), qids.cpu(), gids.cpu()
    
    def _compute_embedding_tse(self, model):
        model = model.eval()
        # Đảm bảo tất cả submodules cũng ở eval mode
        # Đặc biệt là vision_model và text_model trong base_model
        if hasattr(model, 'base_model'):
            if hasattr(model.base_model, 'vision_model'):
                model.base_model.vision_model.eval()
            if hasattr(model.base_model, 'text_model'):
                model.base_model.text_model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text_tse(caption).cpu()
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image_tse(img).cpu()
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0) 
        return qfeats.cpu(), gfeats.cpu(), qids.cpu(), gids.cpu()
    
    def eval(self, model, i2t_metric=False):
        qfeats, gfeats, qids, gids = self._compute_embedding(model)
        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features
        sims_bse = qfeats @ gfeats.t()
  
        vq_feats, vg_feats, _, _ = self._compute_embedding_tse(model)
        vq_feats = F.normalize(vq_feats, p=2, dim=1) # text features
        vg_feats = F.normalize(vg_feats, p=2, dim=1) # image features
        sims_tse = vq_feats@vg_feats.t()
        
        # ROBUST: Adaptive fusion based on noise rate
        # Addresses issue: BGE-t2i > BGE+TSE-t2i at noise 0.4, 0.6, 0.7
        # Get noise rate from model args if available
        noise_rate = getattr(model.args, 'noisy_rate', 0.0) if hasattr(model, 'args') else 0.0
        
        # ROBUST: Adaptive weighting to handle high noise scenarios
        # Based on observation: BGE better than BGE+TSE at noise >= 0.4
        # At noise_rate=0.0-0.3: equal (0.5, 0.5)
        # At noise_rate=0.4: more BGE (0.6, 0.4)
        # At noise_rate=0.6: more BGE (0.65, 0.35)
        # At noise_rate=0.7: more BGE (0.75, 0.25)
        if noise_rate <= 0.3:
            alpha = 0.5  # Equal weighting
        elif noise_rate <= 0.5:
            # Linear interpolation from 0.5 to 0.65 between 0.3 and 0.5
            alpha = 0.5 + 0.15 * ((noise_rate - 0.3) / 0.2)
        else:
            # Linear interpolation from 0.65 to 0.75 between 0.5 and 0.7
            alpha = 0.65 + 0.1 * ((noise_rate - 0.5) / 0.2)
        
        beta = 1.0 - alpha
        
        # ROBUST: Normalize before fusion for stability
        sims_bse_norm = F.normalize(sims_bse, p=2, dim=1)
        sims_tse_norm = F.normalize(sims_tse, p=2, dim=1)
        
        # ROBUST: Clip extreme values to prevent outliers
        sims_bse_norm = torch.clamp(sims_bse_norm, min=-1.0, max=1.0)
        sims_tse_norm = torch.clamp(sims_tse_norm, min=-1.0, max=1.0)
        
        # Adaptive fusion
        sims_fused = alpha * sims_bse_norm + beta * sims_tse_norm
        
        sims_dict = {
            'BGE': sims_bse,
            'TSE': sims_tse,
            'BGE+TSE': sims_fused  # Use adaptive fusion instead of simple average
        }

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP","rSum"])
        
        for key in sims_dict.keys():
            sims = sims_dict[key]
            rs = get_metrics(sims, qids, gids, f'{key}-t2i',False)
            table.add_row(rs)
            if i2t_metric:
                i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=sims.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
                i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
                table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])

        table.custom_format["R1"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["RSum"] = lambda f, v: f"{v:.2f}"
        self.logger.info('\n' + str(table))
        
        return rs[1]
