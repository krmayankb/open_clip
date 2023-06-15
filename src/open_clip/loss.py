import math
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        clip_loss = super().forward(image_features, text_features, logit_scale)
        clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss

class CosRegLoss(ClipLoss):
    def __init__(self, 
                 local_loss=False, 
                 gather_with_grad=False, 
                 cache_labels=False, 
                 rank=0, 
                 world_size=1, 
                 use_horovod=False,
                 cosinereg = 0.001, 
                 reg_threshold = 0.30):
        
        super().__init__(local_loss=local_loss, 
                         gather_with_grad=gather_with_grad, 
                         cache_labels=cache_labels, 
                         rank=rank, 
                         world_size=world_size, 
                         use_horovod=use_horovod)
        
        self.cosinereg = cosinereg
        self.reg_threshold = reg_threshold
    
    def get_modality_cosine_reg(self, features):
        batch_size = features.shape[0]
        
        # calculate dot product of all the representations 
        dot_products = torch.matmul(features, features.t())

        # Exclude the dot products of representation with itself 
        dot_products -= torch.diag(torch.diag(dot_products))
        dot_products = torch.where(dot_products > self.reg_threshold, torch.zeros_like(dot_products), dot_products)
        loss = torch.sum(dot_products)/(batch_size * (batch_size - 1))
        return loss


    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        
        #CLIP Loss 
        cliploss = super().forward(image_features, text_features, logit_scale, output_dict=False)

        #Regularization term on image 
        image_reg_loss = self.get_modality_cosine_reg(image_features)
        
        #Regularization term on text 
        text_reg_loss = self.get_modality_cosine_reg(text_features)

        loss_regularization = self.cosinereg * (image_reg_loss + text_reg_loss)
        
        if output_dict:
            return {"cosine_reg": loss_regularization, "contrastive_loss": cliploss}
        
        return cliploss, loss_regularization

class BYOLCLIPLOSS(ClipLoss):
    def __init__(self, 
                 local_loss=False, 
                 gather_with_grad=False, 
                 cache_labels=False, 
                 rank=0, 
                 world_size=1, 
                 use_horovod=False
                 ):
        super().__init__(local_loss, 
                         gather_with_grad, 
                         cache_labels, 
                         rank, world_size, 
                         use_horovod
                         )
    def forward(self, 
                image_features, 
                text_features, 
                logit_scale, 
                batch_byol_loss, 
                output_dict=False
                ): 
        
        byol_loss = batch_byol_loss * 1000
        cliploss = super().forward(image_features, text_features, logit_scale, output_dict=False) 

        # if byol_loss.grad is None: 
        #     byol_loss.requires_grad_(True)

        if output_dict:
            return {"byol_loss": byol_loss, "contrastive_loss": cliploss}
        
        return cliploss, byol_loss

class CenteredClipLoss(ClipLoss):
    def __init__(self, 
                 local_loss=False, 
                 gather_with_grad=False, 
                 cache_labels=False, 
                 rank=0, 
                 world_size=1, 
                 use_horovod=False
                 ):
        
        super().__init__(local_loss=local_loss, 
                         gather_with_grad=gather_with_grad, 
                         cache_labels=cache_labels, 
                         rank=rank, 
                         world_size=world_size, 
                         use_horovod=use_horovod)

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        
        image_features = image_features - torch.mean(image_features, dim=0)
        text_features = text_features - torch.mean(text_features, dim=0) 
        
        #CLIP Loss 
        cliploss = super().forward(image_features, text_features, logit_scale, output_dict=False)
        
        if output_dict:
            return {"contrastive_loss": cliploss}
        
        return cliploss 
    

# SVD based dimension removal and centering - Cosine Regularization 
class SVDCosRegLoss(ClipLoss):
    def __init__(self, 
                 local_loss=False, 
                 gather_with_grad=False, 
                 cache_labels=False, 
                 rank=0, 
                 world_size=1, 
                 use_horovod=False,
                 svd_cosinereg = 0.01, 
                 apply_normal_dist = False, 
                 normal_dist_var = 0.25):
        
        super().__init__(local_loss=local_loss, 
                         gather_with_grad=gather_with_grad, 
                         cache_labels=cache_labels, 
                         rank=rank, 
                         world_size=world_size, 
                         use_horovod=use_horovod)
        
        self.svd_cosinereg = svd_cosinereg
        self.apply_normal_dist = apply_normal_dist
        self.normal_dist_var = normal_dist_var
    
    def remove_top_d(self, features):
        topk= features.shape[0]//100
        u, s, v = torch.svd(features)
        features = torch.matmul(u[:, topk:], torch.matmul(torch.diag(s[topk:]), v.T[topk:, :]))
        return features

    def remove_mean(self, features, dim=0):
        features = features - torch.mean(features, dim=dim, keepdim=True)
        return features
    
    def process_features(self, features):
        features = self.remove_mean(features, dim=0)
        features = self.remove_top_d(features)
        return features
    
    def apply_normal_distribution(self, matrix, var):
        coefficient = 1 / (var * math.sqrt(2 * math.pi))
        exponent = -0.5 * torch.square(torch.div(matrix, var))
        result = torch.exp(exponent) * coefficient
        return result
    
    def get_modality_cosine_reg(self, features):
        batch_size = features.shape[0]
        
        # calculate dot product of all the representations 
        dot_products = torch.matmul(features, features.t())
        if self.apply_normal_dist: 
            dot_products = self.apply_normal_distribution(dot_products, var=self.normal_dist_var)

        # Exclude the dot products of representation with itself 
        dot_products -= torch.diag(torch.diag(dot_products))
        loss = torch.sum(dot_products)/(batch_size * (batch_size - 1))
        return loss


    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        # print("DEBUG: ", self.svd_cosinereg, self.apply_normal_dist, self.normal_dist_var)
        # print("DEBUG: ", type(self.svd_cosinereg), type(self.apply_normal_dist), type(self.normal_dist_var))
        new_image_features = self.process_features(image_features)
        new_text_features = self.process_features(text_features)

        #CLIP Loss 
        cliploss = super().forward(image_features, text_features, logit_scale, output_dict=False)

        #Regularization term on image 
        image_reg_loss = self.get_modality_cosine_reg(new_image_features) 
        text_reg_loss = self.get_modality_cosine_reg(new_text_features)
        loss_regularization = self.svd_cosinereg * (image_reg_loss + text_reg_loss)
        if output_dict:
            return {"reg_loss": loss_regularization, "contrastive_loss": cliploss}
        
        return cliploss, loss_regularization
    

class DINOCLIPLOSS(ClipLoss):
    def __init__(self, 
                 local_loss=False, 
                 gather_with_grad=False, 
                 cache_labels=False, 
                 rank=0, 
                 world_size=1, 
                 use_horovod=False
                 ):
        super().__init__(local_loss, 
                         gather_with_grad, 
                         cache_labels, 
                         rank, world_size, 
                         use_horovod
                         )
    def forward(self, 
                image_features, 
                text_features, 
                logit_scale, 
                loss, 
                output_dict=False
                ): 
        
        loss = loss * 1000
        cliploss = super().forward(image_features, text_features, logit_scale, output_dict=False) 

        if output_dict:
            return {"dino_loss": loss, "contrastive_loss": cliploss}
        
        return cliploss, loss