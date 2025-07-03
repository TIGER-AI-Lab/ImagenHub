from typing import Optional

import torch
import torch.nn
from einops import rearrange
from torch import nn

from .layers import MLP, TextProjection, TimestepEmbedder, apply_gate, attention


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


def get_norm_layer(norm_layer):
    """
    Get the normalization layer.

    Args:
        norm_layer (str): The type of normalization layer.

    Returns:
        norm_layer (nn.Module): The normalization layer.
    """
    if norm_layer == "layer":
        return nn.LayerNorm
    elif norm_layer == "rms":
        return RMSNorm
    else:
        raise NotImplementedError(f"Norm layer {norm_layer} is not implemented")


def get_activation_layer(act_type):
    """get activation layer

    Args:
        act_type (str): the activation type

    Returns:
        torch.nn.functional: the activation layer
    """
    if act_type == "gelu":
        return lambda: nn.GELU()
    elif act_type == "gelu_tanh":
        return lambda: nn.GELU(approximate="tanh")
    elif act_type == "relu":
        return nn.ReLU
    elif act_type == "silu":
        return nn.SiLU
    else:
        raise ValueError(f"Unknown activation type: {act_type}")

class IndividualTokenRefinerBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        heads_num,
        mlp_width_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        need_CA: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.need_CA = need_CA
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs
        )
        self.self_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.self_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.self_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.self_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs
        )
        act_layer = get_activation_layer(act_type)
        self.mlp = MLP(
            in_channels=hidden_size,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop_rate,
            **factory_kwargs,
        )

        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs),
        )

        if self.need_CA:
            self.cross_attnblock=CrossAttnBlock(hidden_size=hidden_size,
                        heads_num=heads_num,
                        mlp_width_ratio=mlp_width_ratio,
                        mlp_drop_rate=mlp_drop_rate,
                        act_type=act_type,
                        qk_norm=qk_norm,
                        qk_norm_type=qk_norm_type,
                        qkv_bias=qkv_bias,
                        **factory_kwargs,)
        # Zero-initialize the modulation
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,  # timestep_aware_representations + context_aware_representations
        attn_mask: torch.Tensor = None,
        y: torch.Tensor = None,
    ):
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)

        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        # Apply QK-Norm if needed
        q = self.self_attn_q_norm(q).to(v)
        k = self.self_attn_k_norm(k).to(v)

        # Self-Attention
        attn = attention(q, k, v, mode="torch", attn_mask=attn_mask)

        x = x + apply_gate(self.self_attn_proj(attn), gate_msa)
        
        if self.need_CA:
            x = self.cross_attnblock(x, c, attn_mask, y)

        # FFN Layer
        x = x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)

        return x




class CrossAttnBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        heads_num,
        mlp_width_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num

        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs
        )
        self.norm1_2 = nn.LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs
        )
        self.self_attn_q = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )
        self.self_attn_kv = nn.Linear(
            hidden_size, hidden_size*2, bias=qkv_bias, **factory_kwargs
        )
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.self_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.self_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.self_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs
        )
        act_layer = get_activation_layer(act_type)

        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs),
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,  # timestep_aware_representations + context_aware_representations
        attn_mask: torch.Tensor = None,
        y: torch.Tensor=None,
        
    ):
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)

        norm_x = self.norm1(x)
        norm_y = self.norm1_2(y)
        q = self.self_attn_q(norm_x)
        q = rearrange(q, "B L (H D) -> B L H D",  H=self.heads_num)
        kv = self.self_attn_kv(norm_y)
        k, v = rearrange(kv, "B L (K H D) -> K B L H D", K=2, H=self.heads_num)
        # Apply QK-Norm if needed
        q = self.self_attn_q_norm(q).to(v)
        k = self.self_attn_k_norm(k).to(v)

        # Self-Attention
        attn = attention(q, k, v, mode="torch", attn_mask=attn_mask)

        x = x + apply_gate(self.self_attn_proj(attn), gate_msa)

        return x



class IndividualTokenRefiner(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        heads_num,
        depth,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        need_CA:bool=False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):  
        
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.need_CA = need_CA
        self.blocks = nn.ModuleList(
            [
                IndividualTokenRefinerBlock(
                    hidden_size=hidden_size,
                    heads_num=heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    act_type=act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    need_CA=self.need_CA,
                    **factory_kwargs,
                )
                for _ in range(depth)
            ]
        )


    def forward(
        self,
        x: torch.Tensor,
        c: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
        y:torch.Tensor=None,
    ):
        self_attn_mask = None
        if mask is not None:
            batch_size = mask.shape[0]
            seq_len = mask.shape[1]
            mask = mask.to(x.device)
            # batch_size x 1 x seq_len x seq_len
            self_attn_mask_1 = mask.view(batch_size, 1, 1, seq_len).repeat(
                1, 1, seq_len, 1
            )
            # batch_size x 1 x seq_len x seq_len
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            # batch_size x 1 x seq_len x seq_len, 1 for broadcasting of heads_num
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            # avoids self-attention weight being NaN for padding tokens
            self_attn_mask[:, :, :, 0] = True
        
        
        for block in self.blocks:
            x = block(x, c, self_attn_mask,y)

        return x


class SingleTokenRefiner(torch.nn.Module):
    """
    A single token refiner block for llm text embedding refine.
    """
    def __init__(
        self,
        in_channels,
        hidden_size,
        heads_num,
        depth,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        need_CA:bool=False,
        attn_mode: str = "torch",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.attn_mode = attn_mode
        self.need_CA = need_CA
        assert self.attn_mode == "torch", "Only support 'torch' mode for token refiner."

        self.input_embedder = nn.Linear(
            in_channels, hidden_size, bias=True, **factory_kwargs
        )
        if self.need_CA:
            self.input_embedder_CA = nn.Linear(
            in_channels, hidden_size, bias=True, **factory_kwargs
        )

        act_layer = get_activation_layer(act_type)
        # Build timestep embedding layer
        self.t_embedder = TimestepEmbedder(hidden_size, act_layer, **factory_kwargs)
        # Build context embedding layer
        self.c_embedder = TextProjection(
            in_channels, hidden_size, act_layer, **factory_kwargs
        )

        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size=hidden_size,
            heads_num=heads_num,
            depth=depth,
            mlp_width_ratio=mlp_width_ratio,
            mlp_drop_rate=mlp_drop_rate,
            act_type=act_type,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            qkv_bias=qkv_bias,
            need_CA=need_CA,
            **factory_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.LongTensor,
        mask: Optional[torch.LongTensor] = None,
        y: torch.LongTensor=None,
    ):
        timestep_aware_representations = self.t_embedder(t)

        if mask is None:
            context_aware_representations = x.mean(dim=1)
        else:
            mask_float = mask.unsqueeze(-1)  # [b, s1, 1]
            context_aware_representations = (x * mask_float).sum(
                dim=1
            ) / mask_float.sum(dim=1)
        context_aware_representations = self.c_embedder(context_aware_representations)
        c = timestep_aware_representations + context_aware_representations

        x = self.input_embedder(x)
        if self.need_CA:
            y = self.input_embedder_CA(y)
            x = self.individual_token_refiner(x, c, mask, y)
        else:
            x = self.individual_token_refiner(x, c, mask)

        return x



class Qwen2Connector(torch.nn.Module):
    def __init__(
        self,
        in_channels=3584,
        hidden_size=4096,
        heads_num=32,
        depth=2,
        need_CA=False,
        device=None,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype":dtype}

        self.S =SingleTokenRefiner(in_channels=in_channels,hidden_size=hidden_size,heads_num=heads_num,depth=depth,need_CA=need_CA,**factory_kwargs)
        self.global_proj_out=nn.Linear(in_channels,768)

        self.scale_factor = nn.Parameter(torch.zeros(1))
        with torch.no_grad():
            self.scale_factor.data += -(1 - 0.09)

    def forward(self, x,t,mask):
        t = t * 1000 # fix the times embedding bug
        mask_float = mask.unsqueeze(-1)  # [b, s1, 1]
        x_mean = (x * mask_float).sum(
                dim=1
            ) / mask_float.sum(dim=1) * (1 + self.scale_factor.to(x.dtype))

        global_out=self.global_proj_out(x_mean)
        encoder_hidden_states = self.S(x,t,mask)
        return encoder_hidden_states,global_out