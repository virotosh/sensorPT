import math
import torch
import torch.nn as nn

from .module import PatchEmbed, Block, RotaryEmbedding
from .common import CHANNEL_DICT, trunc_normal_, apply_mask, apply_mask_t


class SensorTransformerEncoder(nn.Module):
    """ Sensor Transformer """
    def __init__(
        self,
        img_size=(31,1000),
        patch_size=64,
        patch_stride=None,
        embed_dim=768,
        embed_num=1,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        patch_module=PatchEmbed,
        init_std=0.02,
        interpolate_factor = 2.,
        return_attention_layer=-1,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.embed_num = embed_num
        
        self.num_heads = num_heads
        
        # --
        self.patch_embed = patch_module(
            img_size=img_size,
            patch_size=patch_size,
            patch_stride=patch_stride,
            embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        # --
        
        self.chan_embed = nn.Embedding(len(CHANNEL_DICT), embed_dim)
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                is_causal=False, use_rope= False, return_attention=(i+1)==return_attention_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))
            
        trunc_normal_(self.summary_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()
        
    def prepare_chan_ids(self, channels):
        chan_ids = []
        for ch in channels:
            ch = ch.upper().strip('.')
            assert ch in CHANNEL_DICT
            chan_ids.append(CHANNEL_DICT[ch])
        return torch.tensor(chan_ids).unsqueeze_(0).long()
    
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, chan_ids=None, mask_x=None, mask_t=None):
        # x.shape B, C, T
        # mask_x.shape mN, mC
        # mask_t.shape mN
        
        # -- patchify x
        x = self.patch_embed(x) #
        B, N, C, D = x.shape
        
        assert N==self.num_patches[1] and C==self.num_patches[0], f"{N}=={self.num_patches[1]} and {C}=={self.num_patches[0]}"
        
        if chan_ids is None:
            chan_ids = torch.arange(0,C)     
        chan_ids = chan_ids.to(x)
        
        # -- add channels positional embedding to x
        x = x + self.chan_embed(chan_ids.long()).unsqueeze(0) # (1,C) -> (1,1,C,D)
        
        if mask_x is not None:
            mask_x = mask_x.to(x.device)
            x = apply_mask(mask_x, x)# B, mN, mC, D
            B, N, C, D = x.shape
            
        
        x = x.flatten(0, 1) # BmN, mC, D
        
        # -- concat summary token
        summary_token = self.summary_token.repeat((x.shape[0], 1, 1))
        x = torch.cat([x,summary_token], dim=1)  # BmN, mC+embed_num, D
        
        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x) # B*N, mC+1, D
            if blk.return_attention==True: return x

        x = x[:, -summary_token.shape[1]:, :]
        
        if self.norm is not None:
            x = self.norm(x) 

        
        x = x.flatten(-2)
        x = x.reshape((B, N, -1))
        # -- reshape back
            
        if mask_t is not None:
            mask_t = mask_t.to(x.device)
            x = apply_mask_t(mask_t, x)# B, mN, D        
        
        x = x.reshape((B, N, self.embed_num, -1))
        
        return x


class SensorTransformerReconstructor(nn.Module):
    """ Sensor Transformer """
    def __init__(
        self,
        num_patches,
        patch_size=64,
        embed_num=1,
        use_pos_embed = False,
        use_inp_embed = True,
        embed_dim=768,
        reconstructor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        interpolate_factor = 2.,
        return_attention_layer=-1,
        **kwargs
    ):
        super().__init__()
        self.use_inp_embed = use_inp_embed
        self.use_pos_embed = use_pos_embed
        self.num_patches = num_patches
        
        if use_inp_embed:
            self.reconstructor_embed = nn.Linear(embed_dim, reconstructor_embed_dim, bias=True)
        
        if use_pos_embed:
            self.pos_embed           = nn.Parameter(torch.zeros(1, 1, embed_num, reconstructor_embed_dim))
            trunc_normal_(self.pos_embed, std=init_std)
        
        self.mask_token          = nn.Parameter(torch.zeros(1, 1, reconstructor_embed_dim))
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.time_embed_dim = (reconstructor_embed_dim//num_heads)//2
        self.time_embed = RotaryEmbedding(dim=self.time_embed_dim, interpolate_factor=interpolate_factor)
        self.chan_embed = nn.Embedding(len(CHANNEL_DICT), reconstructor_embed_dim)
        # --
        self.reconstructor_blocks = nn.ModuleList([
            Block(
                dim=reconstructor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, is_causal=False, use_rope=True, 
                return_attention=(i+1)==return_attention_layer)
            for i in range(depth)])
        self.reconstructor_norm = norm_layer(reconstructor_embed_dim)
        self.reconstructor_proj = nn.Linear(reconstructor_embed_dim, patch_size, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()
        

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.reconstructor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def forward(self, x, chan_ids=None, mask_x=None, mask_y=None):
        # conditions: (Nq, D) as qurey for downstream 
        # mask_x/mask_y: (mN, mC) one number index like (n*C+c) in matrix (N,C)
        
        chan_ids = chan_ids.to(x).long()
        
        # -- map from encoder-dim to pedictor-dim
        if self.use_inp_embed:
            x = self.reconstructor_embed(x)

        C, N        = self.num_patches
        B, mN, eN, D= x.shape
        # assert mN == N, f"{mN},{N}"
        ############## Mask x ###############
        # -- add channels positional embedding to x
        chan_embed = self.chan_embed(chan_ids).unsqueeze(0) # (1,C) -> (1,1,C,D)
        # -- get freqs for RoPE
        
        if mask_x is not None:
            mask_x       = mask_x.to(x.device)
            mask_x       = torch.floor(mask_x[:,0] / C).long().to(x.device)    # select first as represent
            
            freqs_x      = self.time_embed.prepare_freqs((1, N), x.device, x.dtype)
            freqs_x      = freqs_x.contiguous().view((1,N,self.time_embed_dim))
            freqs_x      = apply_mask_t(mask_x, freqs_x)                                       # 1, mN, 1, D
            freqs_x      = freqs_x.contiguous().view((mask_x.shape[0], 1, self.time_embed_dim)) # mN, D//2
            freqs_x      = freqs_x.repeat((1, eN, 1)).flatten(0,1)
            
        else:
            freqs_x      = self.time_embed.prepare_freqs((eN, N), x.device, x.dtype) # NC, time_dim
            
        ############# Mask y ################
        if mask_y is not None:
            mask_y       = mask_y.to(x.device)
            
            # create query mask_token ys
            N_y          = mask_y.shape[0]
            chan_embed   = chan_embed.repeat((1,N,1,1))
            chan_embed   = apply_mask(mask_y, chan_embed)
            
            freqs        = self.time_embed.prepare_freqs((C, N), x.device, x.dtype) # NC, time_dim
            freqs_y      = freqs.contiguous().view((1, N, C, self.time_embed_dim))
            freqs_y      = apply_mask(mask_y, freqs_y)        # 1, mN, mC, D
            freqs_y      = freqs_y.contiguous().view((N_y, self.time_embed_dim))
            
            y = self.mask_token.repeat((B, N_y, 1)) + chan_embed
            
            
            if self.use_pos_embed:
                x        = x + self.pos_embed.repeat((B, x.shape[1], 1, 1)).to(x.device)
                
            # -- concat query mask_token ys
            x           = x.flatten(1,2) # B N E D -> B NE D
            x           = torch.cat([x,y], dim=1)
            freqs_x     = torch.cat([freqs_x, freqs_y], dim=0).to(x)
            
            
            # -- fwd prop
            for blk in self.reconstructor_blocks:
                x = blk(x, freqs_x) # B, NC, D
                if blk.return_attention==True: return x
            
            
            x = x[:,-N_y:,:]      # B, N_y, D
            
            x = self.reconstructor_norm(x) 
                
            x = self.reconstructor_proj(x)
            
            return x
        
class SensorTransformerPredictor(nn.Module):
    """ Sensor Transformer """
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        embed_num=1,
        use_pos_embed = False,
        use_inp_embed = True,
        use_part_pred = False,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        interpolate_factor = 2.,
        return_attention_layer=-1,
        **kwargs
    ):
        super().__init__()
        self.use_part_pred = use_part_pred
        self.use_pos_embed = use_pos_embed
        self.use_inp_embed = use_inp_embed
        self.num_patches = num_patches
        self.embed_num = embed_num
        
        if use_inp_embed:
            self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        
        if use_pos_embed:
            self.pos_embed   = nn.Parameter(torch.zeros(1, 1, embed_num, predictor_embed_dim))
            trunc_normal_(self.pos_embed, std=init_std)
        
        self.mask_token      = nn.Parameter(torch.zeros(1, 1, embed_num, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.time_embed_dim = (predictor_embed_dim//num_heads)//2
        self.time_embed = RotaryEmbedding(dim=self.time_embed_dim, interpolate_factor=interpolate_factor)
        
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, is_causal=False, use_rope=True, 
                return_attention=(i+1)==return_attention_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()
        

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, mask_x=None, mask_t=None):
        # conditions: (Nq, D) as qurey for downstream 
        # mask_t: mN one number index like (n*C+c) in matrix (N,1)
        
        # -- map from encoder-dim to pedictor-dim
        if self.use_part_pred:
            inp_x = x
            
        if self.use_inp_embed:
            x = self.predictor_embed(x)

        C, N        = self.num_patches
        B, mN, eN, D    = x.shape
        
        ############## Mask x ###############
        # -- get freqs for RoPE
        freqs = self.time_embed.prepare_freqs((eN, N), x.device, x.dtype) # NC, time_dim
        
        if mask_x is not None:
            mask_x       = mask_x
            mask_x       = torch.floor(mask_x[:,0] / C).long()
            ############# Mask y ################
            if mask_t is None:
                mask_t       = torch.tensor(list(set(list(range(0,N))) - set(mask_x.tolist()))).long()
            # -- concat query mask_token ys
            N_y              = mask_t.shape[0]
            y                = self.mask_token.repeat((B, N_y, 1, 1))
            x                = torch.cat([x,y], dim=1)
            
            # -- masked index of tensor x rearrange to normal index
            mask_id          = torch.concat([mask_x.to(x.device), mask_t.to(x.device)], dim=0)            
            x                = torch.index_select(x, dim=1, index=torch.argsort(mask_id))    
            
        if self.use_pos_embed:
            x                = x + self.pos_embed.repeat((B, x.shape[1], 1, 1)).to(x.device)
            
        B, N, eN, D    = x.shape
        x = x.flatten(1,2)
        
        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x, freqs) # B, NC, D
            if blk.return_attention==True: return x
        
        # -- reshape back
        x = x.reshape((B, N, eN, D))
        
        x = self.predictor_norm(x) 
            
        x = self.predictor_proj(x)
        
        if self.use_part_pred and mask_x is not None:
            cmb_x = torch.index_select(x, dim=1, index=mask_t.to(x.device)) 
            cmb_x = torch.concat([inp_x, cmb_x], dim=1)
            cmb_x = torch.index_select(cmb_x, dim=1, index=torch.argsort(mask_id)) 
            return x, cmb_x
        return x