import math
import torch
import random
import numpy as np

use_channels_names = [      'FP1', 'FPZ', 'FP2', 
                               'AF3', 'AF4', 
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                      'PO7', 'PO3', 'POZ',  'PO4', 'PO8', 
                               'O1', 'OZ', 'O2', ]

CHANNEL_DICT = {k.upper():v for v,k in enumerate(use_channels_names)}

def InfoNCELoss(pred, target, t=0.):
    # B, NC, D
    NC, B, D = pred.shape
    similarity = torch.matmul(pred, target.transpose(1,2)) * math.exp(t)#  NC, B, B
    
    label = torch.arange(B).repeat(repeats=(NC,)).to(pred.device)
    
    logit1 = similarity.view((NC*B, B))
    accuracy1 = ((torch.argmax(logit1, dim=-1)==label)*1.0).mean()
    loss1 = torch.nn.functional.cross_entropy(logit1, label)
    
    logit2 = similarity.transpose(1,2).contiguous().view((NC*B, B))
    accuracy2 = ((torch.argmax(logit2, dim=-1)==label)*1.0).mean()
    loss2 = torch.nn.functional.cross_entropy(logit2, label)
    
    return (loss1+loss2)/2, (accuracy1 + accuracy2)/2

def BatchMAE_InfoNCELoss(pred, target, t=0.):
    # B, N, D
    # Compare space-time dimensions
    pred1    = pred - pred.mean(dim=0)
    target1  = target - target.mean(dim=0)
    loss1, accuracy1 = InfoNCELoss(pred1, target1)
    
    # Compare Batch dimensions
    pred2    = pred.transpose(0,1) # N,B,D
    target2  = target.transpose(0,1) 
    
    loss2, accuracy2 = InfoNCELoss(pred2, target2)
    
    return loss1, loss2, accuracy1, accuracy2

def CoupleInfoNCELoss(pred, target, t=0.):
    
    # Compare Batch dimensions
    pred2    = pred.flatten(1,2).transpose(0,1) # NC,B,D
    target2  = target.flatten(1,2).transpose(0,1) 
    
    loss2, accuracy2 = InfoNCELoss(pred2, target2)
    
    # Compare time dimension
    pred1    = pred.transpose(1,2).flatten(0,1)# BC, N, D
    target1  = target.transpose(1,2).flatten(0,1)
    loss1, accuracy1 = InfoNCELoss(pred1, target1)
    
    # Compare Channel dimensions
    pred3    = pred - pred.mean(dim=0)
    pred3    = pred3.flatten(0,1) # BN, C, D
    # pred3    = pred.flatten(0,1).transpose(0,2) # D, C, BN
    # pred3    = torch.layer_norm(pred3, normalized_shape=pred3.shape[-1:]).transpose(0,2) # BN, C, D
    
    
    # target3  = target.flatten(0,1).transpose(0,2) # D, C, BN
    # target3  = torch.layer_norm(target3, normalized_shape=target3.shape[-1:]).transpose(0,2)
    
    target3  = target - target.mean(dim=0)
    target3  = target3.flatten(0,1)
    
    loss3, accuracy3 = InfoNCELoss(pred3, target3)
    return loss1, loss2, loss3, accuracy1, accuracy2, accuracy3
    
def _generate_negatives(z, num_negatives=20):
    """Generate negative samples to compare each sequence location against"""
    batch_size, feat, full_len = z.shape
    z_k = z.permute([0, 2, 1]).reshape(-1, feat)
    with torch.no_grad():
        # candidates = torch.arange(full_len).unsqueeze(-1).expand(-1, self.num_negatives).flatten()
        negative_inds = torch.randint(0, full_len, size=(batch_size, full_len * num_negatives))
        # From wav2vec 2.0 implementation, I don't understand
        # negative_inds[negative_inds >= candidates] += 1

        for i in range(1, batch_size):
            negative_inds[i] += i * full_len

    z_k = z_k[negative_inds.view(-1)].view(batch_size, full_len, num_negatives, feat)
    return z_k, negative_inds

def _calculate_similarity(z, c, negatives, temp=0.1):
    c = c.permute([0, 2, 1]).unsqueeze(-2) # b, s, 1, t
    z = z.permute([0, 2, 1]).unsqueeze(-2)

    # negatives do not contain conflicting loss items
    negative_in_target = (c == negatives).all(dim=-1) | (z == negatives).all(dim=-1)

    targets = torch.cat([c, negatives], dim=-2)
    # print(targets.shape)
    logits = torch.nn.functional.cosine_similarity(z, targets, dim=-1) / temp

    if negative_in_target.any():
        # print(negative_in_target.shape, logits.shape)
        logits[:,:,1:][negative_in_target] = float("-inf")
    
    return logits.view(-1, logits.shape[-1])

class SelfSuperviseLoss():
    def __init__(self, device='cuda', beta=1.0, num_negatives=10) -> None:
        self.beta = beta
        self.device = device
        self.num_negatives = num_negatives
        self.loss_fn = torch.nn.CrossEntropyLoss().to(device)

    def __call__(self, dec_data, enc_data):
        negatives, _ = _generate_negatives(enc_data, num_negatives=self.num_negatives)
        # Prediction -> batch_size x predict_length x predict_length
        logits = _calculate_similarity(enc_data, dec_data, negatives)
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        return self.loss_fn(logits, labels) + self.beta * enc_data.pow(2).mean(), logits 
    
    

class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats

class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        return new_lr


class CosineWDSchedule(object):

    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.
    ):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd

def seed_torch(seed=1029):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True





def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def apply_mask(mask, x):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), C, D (feature-dim)]
    :param mask: tensor [mN, mC] containing indices of patches in [N, C] to keep 
    """    
    B, N, C, D = x.shape
    if len(mask.shape)==2:
        mN, mC = mask.shape
        
        mask_keep = mask.reshape((1,mN*mC,1)).repeat((B, 1, D))
        masked_x = torch.gather(x.reshape((B, N*C, D)), dim=-2, index=mask_keep)
        masked_x = masked_x.contiguous().view((B,mN,mC,D))
    else:
        mN = mask.shape[0]
        
        mask_keep = mask.reshape((1,mN,1)).repeat((B, 1, D))
        masked_x = torch.gather(x.reshape((B, N*C, D)), dim=-2, index=mask_keep)
    return masked_x

def apply_mask_t(mask_t, x):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), C, D (feature-dim)]
    :param mask: tensor [mN, mC] containing indices of patches in [N, C] to keep 
    """    
    B, N, D = x.shape
    mN = mask_t.shape[0]
    
    mask_keep = mask_t.reshape((1,mN,1)).repeat((B, 1, D))
    masked_x = torch.gather(x, dim=1, index=mask_keep)
    return masked_x

def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x

# helper functions
def exists(val):
    return val is not None
    