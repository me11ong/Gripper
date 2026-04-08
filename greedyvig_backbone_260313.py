## Depth 없이, Greedy Quad로 Graph
## mmdet 3.x 호환 버전

import torch
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
import numpy as np
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath
from timm.models.registry import register_model
import clip
from DepthAnythingWrapper import DepthAnythingDeterministicWrapper


try:
    # mmdet 3.x registry
    from mmdet.registry import MODELS as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmengine.runner import load_checkpoint as _load_checkpoint

    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 30, 'input_size': (3, 384, 384), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'greedyvig': _cfg(crop_pct=1.0, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
}


class DepthEstimation(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_model = DepthAnythingDeterministicWrapper(
            checkpoint_path="/mnt/E/Gripper/참고_코드/GreedyViG/detection/depth_anything_v2_vitb.pth",
            device='cuda',
            seed=1234,
            use_torch_deterministic=False,
            verbose=True
        )
        for param in self.depth_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            self.depth_tensor = torch.randn(x.shape[0], 1, 384, 384)
            for i in range(x.shape[0]):
                img_np = x[i].permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)

                predictions = self.depth_model.infer(img_np)
                predictions = torch.tensor(predictions).unsqueeze(0)

                min_val = predictions.min()
                max_val = predictions.max()
                predictions_norm = (predictions - min_val) / (max_val - min_val)
                self.depth_tensor[i] = predictions_norm

            return self.depth_tensor


class DepthGuidedPatchExpansion(nn.Module):
    def __init__(self, patch_size=16, alpha=5.0):
        super().__init__()
        self.patch_size = patch_size
        self.alpha = alpha

    def forward(self, cost_matrix, depth):
        B, patches, C = cost_matrix.shape
        patch_size = 16
        H = depth.shape[-2]
        W = depth.shape[-1]
        Hp = H // patch_size
        Wp = W // patch_size

        patch_cost = cost_matrix.view(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()
        P = patch_size

        depth_patches = F.unfold(depth, kernel_size=P, stride=P)
        depth_patches = depth_patches.view(B, P * P, Hp, Wp)

        alpha = 5.0
        weights = torch.exp(-alpha * depth_patches)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        pixel_cost = patch_cost.unsqueeze(2) * weights.unsqueeze(1) * (P * P)
        pixel_cost = pixel_cost.view(B, C * P * P, Hp * Wp)
        pixel_cost = F.fold(pixel_cost, output_size=(H, W), kernel_size=P, stride=P)
        pixel_cost_matrix = pixel_cost.permute(0, 2, 3, 1).contiguous()

        return pixel_cost_matrix


class CostMatrix(nn.Module):
    def __init__(self, clip_resolution=(384, 384), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clip_model, _ = clip.load("ViT-B/16")
        self.clip_model = self.clip_model.float()
        self.clip_resolution = clip_resolution
        self.templates = ["{}"]

        # visual encoder freeze
        for param in self.clip_model.visual.parameters():
            param.requires_grad = False

        # text encoder 학습
        for param in self.clip_model.transformer.parameters():
            param.requires_grad = True

        self.max_length = 1
        self.expand_patches = DepthGuidedPatchExpansion()

        self.text_spatial_proj = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 576 * 512),
        )

    def get_text_embeds(self, classnames, templates, clip_model, prompt=None):
        tokenizer = None
        tokens = None

        if tokens is None or prompt is not None:
            tokens = []
            for classname in classnames:
                if ', ' in classname:
                    classname_splits = classname.split(', ')
                    texts = [template.format(classname_splits[0]) for template in templates]
                else:
                    texts = [template.format(classname) for template in templates]

                if tokenizer is not None:
                    texts = tokenizer(texts).cuda()
                else:
                    texts = clip.tokenize(texts).cuda()
                tokens.append(texts)

            tokens = torch.stack(tokens, dim=0).squeeze(1)
            if prompt is None:
                tokens = tokens
        elif tokens is not None and prompt is None:
            tokens = tokens

        class_embeddings = clip_model.encode_text(tokens, prompt)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

        batch_size, embed_dim = class_embeddings.shape

        if batch_size < self.max_length:
            padding = torch.zeros(self.max_length - batch_size, embed_dim,
                                  device=class_embeddings.device, dtype=class_embeddings.dtype)
            class_embeddings = torch.cat([class_embeddings, padding], dim=0)
        elif batch_size > self.max_length:
            class_embeddings = class_embeddings[:self.max_length]

        class_embeddings = class_embeddings.unsqueeze(0)
        return class_embeddings

    def forward(self, x_img, x_text, depth):
        visual_features = self.clip_model.encode_image(x_img, dense=True)[:, 1:, :]  # [B, 576, 512]

        text_features_global = []
        for text in x_text:
            text = list(set(text))
            text_feature = self.get_text_embeds(text, self.templates, self.clip_model)
            text_features_global.append(text_feature.squeeze(0))

        text_features_global = torch.cat(text_features_global, dim=0)  # [B, 512]

        text_features_spatial = self.text_spatial_proj(text_features_global)  # [B, 576*512]
        text_features_spatial = text_features_spatial.view(-1, 576, 512)

        visual_features_norm = F.normalize(visual_features, dim=-1)
        text_features_norm = F.normalize(text_features_spatial, dim=-1)

        cost_matrix = (visual_features_norm * text_features_norm).sum(dim=-1, keepdim=True)

        expanded_cost_matrix = self.expand_patches(cost_matrix, depth)
        expanded_cost_matrix = expanded_cost_matrix.permute(0, 3, 1, 2)

        combined_features = torch.cat([x_img, expanded_cost_matrix], dim=1)

        return text_features_global, combined_features


class DepthBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, dilation, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=stride, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class TextGuidedFusion(nn.Module):
    """
    Text-Guided Feature Fusion for FPN outputs
    Fuses text embeddings with spatial features using FiLM (Feature-wise Linear Modulation)
    """
    def __init__(self, feat_dim=256, text_dim=512):
        super(TextGuidedFusion, self).__init__()
        self.modulation = nn.Sequential(
            nn.Linear(text_dim, feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim * 2, feat_dim * 2)
        )

    def forward(self, spatial_features, text_embedding):
        B, C, H, W = spatial_features.shape
        modulation_params = self.modulation(text_embedding)
        gamma, beta = modulation_params.chunk(2, dim=-1)
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        modulated_features = gamma * spatial_features + beta
        return modulated_features


class Stem(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.stem(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.Encoder(x)


class DepthWiseSeparable(nn.Module):
    def __init__(self, in_dim, kernel, expansion=4):
        super().__init__()
        self.pw1 = nn.Conv2d(in_dim, in_dim * 4, 1)
        self.norm1 = nn.BatchNorm2d(in_dim * 4)
        self.act1 = nn.GELU()
        self.dw = nn.Conv2d(in_dim * 4, in_dim * 4, kernel_size=kernel, stride=1, padding=1, groups=in_dim * 4)
        self.norm2 = nn.BatchNorm2d(in_dim * 4)
        self.act2 = nn.GELU()
        self.pw2 = nn.Conv2d(in_dim * 4, in_dim, 1)
        self.norm3 = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        x = self.pw1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dw(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.pw2(x)
        x = self.norm3(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, dim, kernel, expansion_ratio=4., drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        self.dws = DepthWiseSeparable(in_dim=dim, kernel=kernel, expansion=expansion_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.dws(x))
        else:
            x = x + self.drop_path(self.dws(x))
        return x


class QuadMRConv4d(nn.Module):
    def __init__(self, in_channels, out_channels, quad, quad_shift, tau=1.0):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.tau = tau
        self.quad = quad
        self.quad_shift = quad_shift

    def roll_and_connect(self, x, x_rolls):
        B, C, H, W = x.shape
        all_dists = []
        for x_rolled in x_rolls:
            dist = torch.norm(x - x_rolled, p=1, dim=1, keepdim=True)
            all_dists.append(dist)

        all_dists_cat = torch.cat(all_dists, dim=1)
        mean = all_dists_cat.mean(dim=(1, 2, 3), keepdim=True)
        std = all_dists_cat.std(dim=(1, 2, 3), keepdim=True)

        x_j = torch.zeros_like(x)
        weight_sum = torch.zeros_like(all_dists[0])

        for dist, x_rolled in zip(all_dists, x_rolls):
            mask = torch.where(dist < mean - std, 1, 0)
            weight = torch.exp(-dist / self.tau) * mask
            x_j += weight * (x_rolled - x)
            weight_sum += weight

        return x_j, weight_sum

    def forward(self, x):
        B, C, H, W = x.shape
        global_xj = torch.zeros_like(x)
        global_wsum = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)

        local_rolls = [
            torch.roll(x, shifts=(-1, -1), dims=(2, 3)),
            torch.roll(x, shifts=(+1, -1), dims=(2, 3)),
            torch.roll(x, shifts=(0, +1), dims=(2, 3)),
            torch.roll(x, shifts=(+1, 0), dims=(2, 3)),
        ]
        xj_local, w_local = self.roll_and_connect(x, local_rolls)
        global_xj += xj_local
        global_wsum += w_local

        if self.quad:
            quad_rolls = [
                torch.roll(x, shifts=(-H // 2, -W // 2), dims=(2, 3)),
                torch.roll(x, shifts=(+H // 2, -W // 2), dims=(2, 3)),
                torch.roll(x, shifts=(0, +W // 2), dims=(2, 3)),
                torch.roll(x, shifts=(+H // 2, 0), dims=(2, 3)),
            ]
            xj_quad, w_quad = self.roll_and_connect(x, quad_rolls)
            global_xj += xj_quad
            global_wsum += w_quad

        if self.quad_shift:
            quad_shift_rolls = [
                torch.roll(x, shifts=(-H // 4, -W // 4), dims=(2, 3)),
                torch.roll(x, shifts=(+H // 4, -W // 4), dims=(2, 3)),
                torch.roll(x, shifts=(0, +W // 4), dims=(2, 3)),
                torch.roll(x, shifts=(+H // 4, 0), dims=(2, 3)),
            ]
            xj_shift, w_shift = self.roll_and_connect(x, quad_shift_rolls)
            global_xj += xj_shift
            global_wsum += w_shift

        x_j = global_xj / (global_wsum + 1e-6)
        x_cat = torch.cat([x, x_j], dim=1)
        out = self.nn(x_cat)
        return out


class ConditionalPositionEncoding(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.pe = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
            bias=True, groups=in_channels
        )

    def forward(self, x):
        x = self.pe(x) + x
        return x


class Grapher(nn.Module):
    def __init__(self, in_channels, quad, quad_shift):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.cpe = ConditionalPositionEncoding(in_channels, kernel_size=7)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = QuadMRConv4d(in_channels * 2, in_channels, quad, quad_shift)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        x = self.cpe(x)
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        return x


class DynamicGraphConvBlock(nn.Module):
    def __init__(self, in_dim, quad, quad_shift, drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        self.mixer = Grapher(in_dim, quad, quad_shift)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim * 4),
            nn.GELU(),
            nn.Conv2d(in_dim * 4, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(in_dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(in_dim), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.mixer(x))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.ffn(x))
        else:
            x = x + self.drop_path(self.mixer(x))
            x = x + self.drop_path(self.ffn(x))
        return x


class Downsample(nn.Module):
    def __init__(self, in_dim, input_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, input_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(input_dim),
        )

    def forward(self, x):
        return self.conv(x)


class GreedyViG_CostMatrix(torch.nn.Module):
    def __init__(self, blocks, channels, kernels, stride,
                 act_func, dropout=0., drop_path=0., emb_dims=512,
                 K=None, distillation=True, num_classes=1000,
                 pretrained=None, out_indices=None):
        super(GreedyViG_CostMatrix, self).__init__()
        self.distillation = distillation
        self.out_indices = out_indices
        self.pretrained = pretrained
        self.stage_names = ['stem', 'local_1', 'local_2', 'local_3', 'global']

        n_blocks = sum([sum(x) for x in blocks])
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]
        dpr_idx = 0

        self.DepthEstimation = DepthEstimation()
        self.costmatrix = CostMatrix(clip_resolution=(384, 384))
        self.stem = Stem(input_dim=4, output_dim=channels[0])

        self.quad = [False, False, True, True]
        self.quad_shift = [False, False, True, True]

        self.backbone = []
        for i in range(len(blocks)):
            stage = []
            local_stages = blocks[i][0]
            global_stages = blocks[i][1]
            if i > 0:
                stage.append(Downsample(channels[i - 1], channels[i]))
            for _ in range(local_stages):
                stage.append(InvertedResidual(dim=channels[i], kernel=3, expansion_ratio=4, drop_path=dpr[dpr_idx]))
                dpr_idx += 1
            for _ in range(global_stages):
                stage.append(DynamicGraphConvBlock(channels[i], quad=self.quad[i], quad_shift=self.quad_shift[i], drop_path=dpr[dpr_idx]))
                dpr_idx += 1
            self.backbone.append(nn.Sequential(*stage))

        self.backbone = nn.Sequential(*self.backbone)
        self.init_weights()
        self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def init_weights(self):
        if self.pretrained:
            print("Pretrained weights being loaded")
            ckpt = _load_checkpoint(self.pretrained, map_location='cpu')
            print("ckpt keys: ", ckpt.keys())
            missing_keys, unexpected_keys = self.load_state_dict(ckpt, False)
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)
        else:
            print("Initializing weights from scratch")
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs, gt_texts):
        depths = self.DepthEstimation(inputs)
        depths = depths.to(inputs.device)

        text_embeddings_global, x = self.costmatrix(x_img=inputs, x_text=gt_texts, depth=depths)
        self.text_embeddings = text_embeddings_global

        x = self.stem(x)

        outs = []
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if i in self.out_indices:
                outs.append(x)

        return outs


if has_mmdet:
    @det_BACKBONES.register_module()
    def greedyvig_s_costmatrix(pretrained=False, **kwargs):
        model = GreedyViG_CostMatrix(
            blocks=[[4, 4], [4, 4], [12, 4], [3, 3]],
            channels=[64, 128, 256, 512],
            kernels=3,
            stride=1,
            act_func='gelu',
            dropout=0.,
            drop_path=0.1,
            emb_dims=768,
            K=[8, 4, 2, 1],
            distillation=True,
            num_classes=30,
            out_indices=[0, 1, 2, 3],
            pretrained=None
        )
        model.default_cfg = default_cfgs['greedyvig']
        return model


# ============================================================
# Text-Guided FPN (Custom FPN with Text Fusion)
# ============================================================

try:
    from mmdet.registry import MODELS as NECKS
    from mmdet.models.necks.fpn import FPN

    @NECKS.register_module()
    class TextGuidedFPN(FPN):
        """
        FPN with Text-Guided Feature Fusion
        Inherits from standard FPN and adds text modulation to each feature level.
        """
        def __init__(self, text_dim=512, *args, **kwargs):
            super(TextGuidedFPN, self).__init__(*args, **kwargs)
            self.text_dim = text_dim
            out_channels = kwargs.get('out_channels', 256)
            num_outs = kwargs.get('num_outs', 5)
            self.text_fusions = nn.ModuleList([
                TextGuidedFusion(feat_dim=out_channels, text_dim=text_dim)
                for _ in range(num_outs)
            ])
            self.text_embeddings = None

        def set_text_embeddings(self, text_embeddings):
            self.text_embeddings = text_embeddings

        def forward(self, inputs):
            fpn_outputs = super(TextGuidedFPN, self).forward(inputs)
            if self.text_embeddings is not None:
                fused_outputs = []
                for i, feat in enumerate(fpn_outputs):
                    fused_feat = self.text_fusions[i](feat, self.text_embeddings)
                    fused_outputs.append(fused_feat)
                return tuple(fused_outputs)
            else:
                return fpn_outputs

except ImportError:
    print("TextGuidedFPN requires mmdet to be installed")
    pass