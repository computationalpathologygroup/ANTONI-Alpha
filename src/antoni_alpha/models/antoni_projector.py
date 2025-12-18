import torch
from torch import nn
import torch.nn.functional as F


class FFN(nn.Module):
    """Simple feed-forward network with GELU activation."""

    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(self.act(self.w1(x))))


class AntoniProjector(nn.Module):
    """
    Projects vision embeddings to LLM embedding space using attentional pooling
    with learnable queries. Uses Q/K in vision_embedding_dim and V in llm_hidden_size
    to eliminate the need for an MLP projection layer.
    """

    def __init__(
        self,
        vision_embedding_dim,
        llm_hidden_size,
        num_output_tokens=256,
        num_query_heads=8,
        num_kv_heads=8,
        dropout=0.0,
        ffn_hidden_dim=None,
    ):
        super().__init__()

        self.vision_embedding_dim = vision_embedding_dim
        self.llm_hidden_size = llm_hidden_size
        self.num_output_tokens = num_output_tokens
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout

        # Validate GQA configuration
        assert num_query_heads % num_kv_heads == 0, (
            f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
        assert vision_embedding_dim % num_query_heads == 0, (
            f"vision_embedding_dim ({vision_embedding_dim}) must be divisible by num_query_heads ({num_query_heads})"
        )
        assert llm_hidden_size % num_query_heads == 0, (
            f"llm_hidden_size ({llm_hidden_size}) must be divisible by num_query_heads ({num_query_heads})"
        )

        self.head_dim_qk = vision_embedding_dim // num_query_heads
        self.head_dim_v = llm_hidden_size // num_query_heads
        self.num_groups = num_query_heads // num_kv_heads

        # Learnable query tokens (in vision embedding space)
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_output_tokens, vision_embedding_dim)
        )
        nn.init.xavier_uniform_(self.query_tokens)

        # Q, K, V projections
        self.q_proj = nn.Linear(vision_embedding_dim, vision_embedding_dim, bias=False)
        self.k_proj = nn.Linear(
            vision_embedding_dim, self.num_kv_heads * self.head_dim_qk, bias=False
        )
        self.v_proj = nn.Linear(
            vision_embedding_dim, self.num_kv_heads * self.head_dim_v, bias=False
        )

        # Output projection
        self.out_proj = nn.Linear(llm_hidden_size, llm_hidden_size, bias=False)

        # Optional FFN
        if ffn_hidden_dim is not None:
            self.ffn = FFN(llm_hidden_size, ffn_hidden_dim, dropout=dropout)
        else:
            self.ffn = None

        # Layer norm
        self.layer_norm = nn.LayerNorm(llm_hidden_size)

    def forward(self, slide_latents, attention_mask=None):
        """
        Args:
            slide_latents: (B, N, D_vision) - Input vision features
            attention_mask: (B, N) - Optional mask, 1 for valid, 0 for masked

        Returns:
            (B, num_output_tokens, D_llm) - Projected embeddings for LLM
        """
        B, N, D_vision = slide_latents.shape

        # Expand query tokens for the batch
        queries = self.query_tokens.expand(B, -1, -1)  # (B, R, D_vision)
        R = queries.size(1)

        # Project Q, K, V
        Q = self.q_proj(queries)  # (B, R, D_vision)
        K = self.k_proj(slide_latents)  # (B, N, num_kv_heads * head_dim_qk)
        V = self.v_proj(slide_latents)  # (B, N, num_kv_heads * head_dim_v)

        # Reshape for multi-head attention
        Q = Q.view(B, R, self.num_query_heads, self.head_dim_qk).transpose(
            1, 2
        )  # (B, num_query_heads, R, head_dim_qk)
        K = K.view(B, N, self.num_kv_heads, self.head_dim_qk).transpose(
            1, 2
        )  # (B, num_kv_heads, N, head_dim_qk)
        V = V.view(B, N, self.num_kv_heads, self.head_dim_v).transpose(
            1, 2
        )  # (B, num_kv_heads, N, head_dim_v)

        # Expand K and V to match number of query heads (GQA)
        if self.num_query_heads != self.num_kv_heads:
            # Repeat each KV head num_groups times
            K = K.repeat_interleave(
                self.num_groups, dim=1
            )  # (B, num_query_heads, N, head_dim_qk)
            V = V.repeat_interleave(
                self.num_groups, dim=1
            )  # (B, num_query_heads, N, head_dim_v)

        # Prepare attention mask for SDPA
        attn_mask = None
        if attention_mask is not None:
            # SDPA expects (B, num_heads, R, N) or broadcastable shape
            # Convert from (B, N) where 1=valid, 0=masked
            # to (B, 1, 1, N) where True=masked, False=valid
            attn_mask = (
                ~attention_mask.bool()
            )  # Invert: 0->True (masked), 1->False (valid)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)

        # Apply scaled dot-product attention
        # Q: (B, num_query_heads, R, head_dim_qk)
        # K: (B, num_query_heads, N, head_dim_qk)
        # V: (B, num_query_heads, N, head_dim_v)
        attn_output = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # (B, num_query_heads, R, head_dim_v)

        # Reshape back
        attn_output = attn_output.transpose(
            1, 2
        ).contiguous()  # (B, R, num_query_heads, head_dim_v)
        attn_output = attn_output.view(B, R, self.llm_hidden_size)  # (B, R, D_llm)

        # Output projection
        output = self.out_proj(attn_output)

        # Optional FFN
        if self.ffn is not None:
            output = output + self.ffn(output)  # Residual connection

        # Layer norm
        output = self.layer_norm(output)

        return output
