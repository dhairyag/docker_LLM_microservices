import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class SmolLM2Config:
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    hidden_act: str = "silu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1e-5
    vocab_size: int = 49152
    rope_theta: float = 100000.0
    use_cache: bool = True
    tie_word_embeddings: bool = True
    head_dim: int = 64
    attention_dropout: float = 0.0
    attention_bias: bool = False
    mlp_bias: bool = False
    model_type: str = "llama"
    torch_dtype: str = "float32"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def _precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (cos + i*sin)"""
    # Only compute frequencies for half the dimension
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)  # [seq_len, dim//2]
    
    # Compute cos and sin
    freqs_cos = torch.cos(freqs)  # [seq_len, dim//2]
    freqs_sin = torch.sin(freqs)  # [seq_len, dim//2]
    
    # Stack real and imaginary parts
    freqs_cis = torch.stack([freqs_cos, freqs_sin], dim=-1)  # [seq_len, dim//2, 2]
    
    return freqs_cis


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to q and k tensors"""
    # Ensure freqs_cis has the right sequence length and is on the right device
    freqs_cis = freqs_cis[:q.shape[1]].to(device=q.device, dtype=q.dtype)
    
    # Extract cos and sin components and reshape them
    # q and k have shape [batch, seq_len, num_heads, head_dim]
    # We need to reshape freqs_cis to match head_dim
    
    # Reshape q and k to split the head_dim into two halves
    q_split = q.unflatten(-1, (2, -1))  # [batch, seq_len, num_heads, 2, head_dim//2]
    k_split = k.unflatten(-1, (2, -1))
    
    cos = freqs_cis[..., 0]  # [seq_len, head_dim//2]
    sin = freqs_cis[..., 1]  # [seq_len, head_dim//2]
    
    # Add dimensions for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
    sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
    
    # Apply rotary embeddings
    q_out = torch.cat([
        q_split[..., 0, :] * cos - q_split[..., 1, :] * sin,
        q_split[..., 1, :] * cos + q_split[..., 0, :] * sin,
    ], dim=-1)
    
    k_out = torch.cat([
        k_split[..., 0, :] * cos - k_split[..., 1, :] * sin,
        k_split[..., 1, :] * cos + k_split[..., 0, :] * sin,
    ], dim=-1)
    
    return q_out, k_out


class SmolLM2Attention(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        
        self.attention_dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_length, _ = hidden_states.shape
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)

        if freqs_cis is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, freqs_cis)

        key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=2)
        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=2)

        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
            )
            attn_output = attn_output.transpose(1, 2)
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            if self.attention_dropout > 0.0 and self.training:
                attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
            attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class SmolLM2MLP(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SmolLM2Block(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = SmolLM2Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SmolLM2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask, freqs_cis)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SmolLM2Model(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        self.dtype = getattr(torch, config.torch_dtype) if hasattr(torch, config.torch_dtype) else torch.float32
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([SmolLM2Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.freqs_cis = _precompute_freqs_cis(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )
        
        self.apply(self._init_weights)
        
        self.to(self.dtype)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        freqs_cis = self.freqs_cis.to(device=hidden_states.device, dtype=hidden_states.dtype)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, freqs_cis)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class SmolLM2ForCausalLM(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        self.model = SmolLM2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return logits, loss 