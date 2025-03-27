import torch
import torch.nn as nn
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class Conv2dWithConstraint(nn.Conv2d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class Conv1dWithConstraint(nn.Conv1d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


def SMMDL_marginal(Cs,Ct):

    '''
    The SMMDL used in the CRGNet.
    Arg:
        Cs:The source input which shape is NxdXd.
        Ct:The target input which shape is Nxdxd.
    '''
    
    Cs = torch.mean(Cs,dim=0)
    Ct = torch.mean(Ct,dim=0)
    
    # loss = torch.mean((Cs-Ct)**2)
    loss = torch.mean(torch.mul((Cs-Ct), (Cs-Ct)))
    
    return loss

def SMMDL_conditional(Cs,s_label,Ct,t_label):
  
    '''
    The Conditional SMMDL of the source and target data.
    Arg:
        Cs:The source input which shape is NxdXd.
        s_label:The label of Cs data.
        Ct:The target input which shape is Nxdxd.
        t_label:The label of Ct data.
    '''
    s_label = s_label.reshape(-1)
    t_label = t_label.reshape(-1)
    
    class_unique = torch.unique(s_label)
    
    class_num = len(class_unique)
    all_loss = 0.0
    
    for c in class_unique:
        s_index = (s_label == c)
        t_index = (t_label == c)
        # print(t_index)
        if torch.sum(t_index)==0:
            class_num-=1
            continue
        c_Cs = Cs[s_index]
        c_Ct = Ct[t_index]
        m_Cs = torch.mean(c_Cs,dim = 0)
        m_Ct = torch.mean(c_Ct,dim = 0)
        loss = torch.mean((m_Cs-m_Ct)**2)
        all_loss +=loss
        
    if class_num == 0:
        return 0
    
    return all_loss/class_num   



# rotary embedding helper functions
def rotate_half(x):
    # x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x = x.reshape((*x.shape[:-1],x.shape[-1]//2, 2))
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    # return rearrange(x, '... d r -> ... (d r)')
    return x.flatten(-2)

def apply_rotary_emb(freqs, t, start_index=0, scale=1.):
    """
    Apply rotary positional embeddings to a tensor.

    The rotary embedding rotates each dimension of the input tensor `t`
    based on the corresponding frequency in `freqs`, using cosine and sine functions. 
    This rotation helps the model preserve positional information.

    Parameters:
    - freqs (Tensor): The frequency embeddings (sine and cosine values precomputed).
    - t (Tensor): The input tensor to which the rotary embeddings are applied.
    - start_index (int): Start index where the rotation will begin within the tensor `t`.
    - scale (float): Scaling factor for the rotation applied.

    Returns:
    - Tensor: The tensor `t` after rotary positional embeddings have been applied.
    """

    freqs = freqs.to(t.device)

    rot_dim = freqs.shape[-1]

    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    t_left, t_middle, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]

    # Apply rotary embeddings to the middle segment.
    t_rotated_middle = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)

    return torch.cat((t_left, t_rotated_middle, t_right), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000, learned_freq=False, interpolate_factor=1.0):
        """
        Rotary Positional Embedding module to encode sequential information into embeddings.
        
        Parameters:
            dim (int): Dimension of the frequency embedding.
            theta (float): A hyperparameter that influences the scale of the frequency embedding.
            learned_freq (bool): Whether the frequencies are learnable parameters.
            interpolate_factor (float): Scaling factor for interpolated positional encoding.
        """
        super().__init__()
        assert interpolate_factor >= 1.0, "Interpolate factor must be >= 1.0"
        
        # Initialize frequency parameters
        self.freqs = nn.Parameter(
            1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim)),
            requires_grad = learned_freq)
        
        self.interpolate_factor = interpolate_factor
        self.cache = {}

    def prepare_freqs(self, num_patches, device='cuda', dtype=torch.float32, offset=0):
        """
        Prepares the frequency embeddings for the given number of patches.
        
        Parameters:
            num_patches (tuple): Tuple specifying the dimensions (C, N) where
                                 C is the channels and N is the number of positions.
            device (str): Device to store the frequencies on (e.g., 'cuda' or 'cpu').
            dtype (torch.dtype): Data type for the frequencies.
            offset (float): Offset added to position indexes before scaling.
            
        Returns:
            torch.Tensor: Prepared frequency embeddings with shape [C * N, dim].
        """
        C, N = num_patches
        cache_key = f'freqs:{num_patches}'
        
        # Return cached result if available
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate sequence positions and apply offset and scale
        seq_pos = torch.arange(N, device=device, dtype=dtype).repeat_interleave(repeats=C)
        seq_pos = (seq_pos + offset) / self.interpolate_factor
        
        # Compute outer product of positions and frequencies, then expand along the last dimension
        freqs_scaled = torch.outer(seq_pos.type(self.freqs.dtype), self.freqs).repeat_interleave(repeats=2, dim=-1)
        
        # Cache and return the computed frequencies
        self.cache[cache_key] = freqs_scaled
        return freqs_scaled


# modules for sensor transformer
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
    
    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., is_causal=False, use_rope=False, return_attention=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.use_rope = use_rope
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal
        self.return_attention= return_attention

    def forward(self, x, freqs=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3,B,nh,t,d
        q, k, v = qkv[0], qkv[1], qkv[2] # B,nh,t,d
        
        if self.use_rope:# RoPE
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)
        if self.return_attention:
            if self.is_causal:
                attn_mask = torch.ones(q.size(-2), q.size(-2), dtype=torch.bool).tril(diagonal=0)
                attn_maak = torch.zeros(q.size(-2), q.size(-2))
                attn_mask = attn_maak.masked_fill(torch.logical_not(attn_mask), -float('inf'))
                attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask, dim=-1)
            else:
                attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1)
            return attn_weight
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=self.is_causal)
        x = y.transpose(1, 2).contiguous().view(B, T, C) #(B, nh, T, hs) -> (B, T, hs*nh)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, is_causal=False, use_rope=False, return_attention=False):
        super().__init__()
        
        self.return_attention= return_attention
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, is_causal=is_causal, use_rope=use_rope, return_attention = return_attention)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, freqs=None):
        y = self.attn(self.norm1(x), freqs)
        if self.return_attention: return y
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(64, 1000), patch_size=16, patch_stride=None, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if patch_stride is None:
            self.num_patches = ((img_size[0]), (img_size[1] // patch_size))
        else:
            self.num_patches = ((img_size[0]), ((img_size[1] - patch_size) // patch_stride + 1))

        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(1,patch_size), 
                              stride=(1, patch_size if patch_stride is None else patch_stride))
        
    def forward(self, x):
        # x: B,C,T
        x = x.unsqueeze(1)# B, 1, C, T
        x = self.proj(x).transpose(1,3) # B, T, C, D
        return x

