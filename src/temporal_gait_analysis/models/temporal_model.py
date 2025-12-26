# # src/temporal_gait_analysis/models/temporal_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# base components
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = None
        if in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_c)
            )
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return self.relu(out)

class CNNBackbone(nn.Module):
    def __init__(self):
        super(CNNBackbone, self).__init__()
        self.in_conv = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.layer1 = BasicBlock(32, 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.layer2 = BasicBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.layer3 = BasicBlock(64, 128)
        self.layer4 = BasicBlock(128, 128)
    def forward(self, x):
        x = self.in_conv(x)
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class TunablePartExtractor(nn.Module):
    def __init__(self, part_config):
        super(TunablePartExtractor, self).__init__()
        self.part_config = part_config
        self.p = nn.Parameter(torch.ones(1) * 3.0)

        if part_config == "corpse":
            self.output_parts = 3 
        elif isinstance(part_config, int):
            self.output_parts = part_config
            self.internal_splits = part_config
        else:
            raise ValueError(f"Configurazione parti non supportata: {part_config}")

    def forward(self, x):
        # x shape: [bt, c, h, w]
        N, C, H, W = x.shape
        
        # gem power
        x_pow = x.clamp(min=1e-6).pow(self.p)
        
        if self.part_config == "corpse":
            # corpse mode
            h_step = H // 4
            if h_step == 0: h_step = 1
            
            # define the three blocks
            feat_head = x_pow[:, :, :h_step, :]             # Testa
            feat_body = x_pow[:, :, h_step : 3*h_step, :]   # Busto (doppia altezza)
            feat_legs = x_pow[:, :, 3*h_step:, :]           # Gambe
            
            # pool each block (output: [bt, c, 1, 1])
            p_head = F.adaptive_avg_pool2d(feat_head, (1, 1))
            p_body = F.adaptive_avg_pool2d(feat_body, (1, 1))
            p_legs = F.adaptive_avg_pool2d(feat_legs, (1, 1))
            
            # input list: 3 tensors [bt, c, 1, 1] -> cat -> [bt, c, 3, 1]
            out = torch.cat([p_head, p_body, p_legs], dim=2)
            
        else:
            # standard mode
            # output shape: [bt, c, num_splits, 1]
            x_pool = F.adaptive_avg_pool2d(x_pow, (self.internal_splits, 1)) 
            out = x_pool

        # out here is [bt, c, num_parts, 1]
        
        out = out.pow(1./self.p)
        # out becomes [bt, c, num_parts]
        out = out.squeeze(3) 
        
        # permute to [bt, num_parts, c]
        out = out.permute(0, 2, 1) 
        
        return out

# relative position bias (rpe)
class RelativePositionBias(nn.Module):
    def __init__(self, max_len=64):
        super().__init__()
        self.max_len = max_len
        # lookup table for relative biases in [-max_len, max_len]
        self.rel_embedding = nn.Parameter(torch.randn(2 * max_len + 1))

    def forward(self, T):
        # build a (t, t) matrix of relative indices
        range_vec = torch.arange(T, device=self.rel_embedding.device)
        # distance matrix: (t, 1) - (1, t) -> (t, t)
        # values range from -(t-1) to +(t-1)
        relative_idx = range_vec[:, None] - range_vec[None, :] 
        # shift to positive indices in [0, 2*max_len]
        relative_idx = relative_idx + self.max_len
        relative_idx = torch.clamp(relative_idx, 0, 2 * self.max_len)
        
        # lookup
        bias = self.rel_embedding[relative_idx] # (T, T)
        return bias

# tunable transformer
class TunableTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, encoding="absolute"):
        super().__init__()
        self.encoding = encoding
        self.d_model = d_model
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # encoding selection
        self.pos_embed = None
        self.cpe_conv = None
        self.cycle_embed = None
        self.rpe_module = None
        
        if encoding == "absolute":
            self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model))
            
        elif encoding == "cpe":
            self.cpe_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
            
        elif encoding == "cycle":
            self.cycle_len = 7 
            self.cycle_embed = nn.Parameter(torch.randn(1, self.cycle_len, d_model))
            
        elif encoding == "sinusoidal":
            self.register_buffer("sin_pe", self._get_sinusoidal_pe(64, d_model))
            
        elif encoding == "rpe":
            self.rpe_module = RelativePositionBias(max_len=64)
            
        else:
            raise ValueError(f"Encoding {encoding} non supportato")

    def _get_sinusoidal_pe(self, length, d_model):
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        # x shape: [b, t, c]
        B, T, C = x.shape
        mask = None
        
        # apply positional encoding
        if self.encoding == "absolute":
            if T <= 64: x = x + self.pos_embed[:, :T, :]
                
        elif self.encoding == "cpe":
            pos_feat = self.cpe_conv(x.transpose(1, 2)).transpose(1, 2)
            x = x + pos_feat
            
        elif self.encoding == "cycle":
            cycle_emb = self.cycle_embed.repeat(1, (T // self.cycle_len) + 1, 1)
            x = x + cycle_emb[:, :T, :]
            
        elif self.encoding == "sinusoidal":
            if T <= 64: x = x + self.sin_pe[:, :T, :]
            
        elif self.encoding == "rpe":
            # build the rpe mask passed to the transformer

            # bias matrix for frames (t, t)
            rpe_bias = self.rpe_module(T) 
            
            # extend the bias to include the cls row and column as zeros
            full_mask = torch.zeros((T+1, T+1), device=x.device, dtype=x.dtype)
            full_mask[1:, 1:] = rpe_bias
            
            # pytorch transformer mask: added to attention logits (float)
            mask = full_mask

        # prepend the cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # [B, T+1, C]
        out = self.transformer(x, mask=mask)
        return out[:, 0, :] 

# main model
class TemporalGaitModel(nn.Module):
    def __init__(self, num_classes=74, part_config=4, encoding="absolute"):
        super(TemporalGaitModel, self).__init__()
        
        self.part_config = part_config
        self.encoding_type = encoding
        self.feat_dim = 128
        self.backbone = CNNBackbone()
        self.part_extractor = TunablePartExtractor(part_config)
        self.num_parts = self.part_extractor.output_parts 
        self.transformer = TunableTransformer(d_model=self.feat_dim, num_layers=2, encoding=encoding)
        
        self.classifiers = nn.ModuleList([nn.Linear(self.feat_dim, num_classes) for _ in range(self.num_parts)])
        self.bn_necks = nn.ModuleList([nn.BatchNorm1d(self.feat_dim) for _ in range(self.num_parts)])
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feat_maps = self.backbone(x)
        parts = self.part_extractor(feat_maps) 
        parts = parts.view(B, T, self.num_parts, self.feat_dim)
        
        part_embeddings = []
        part_logits = []
        
        for i in range(self.num_parts):
            seq_part = parts[:, :, i, :]
            emb_part = self.transformer(seq_part)
            emb_bn = self.bn_necks[i](emb_part)
            part_embeddings.append(emb_bn)
            
            if self.training:
                log_part = self.classifiers[i](emb_bn)
                part_logits.append(log_part)
        
        if self.training: return part_embeddings, part_logits
        else: return F.normalize(torch.cat(part_embeddings, dim=1), p=2, dim=1)
