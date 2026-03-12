import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# ---------------------------------------------------------
# Dynamic Hardware Routing for Mamba
# ---------------------------------------------------------
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("WARNING: mamba_ssm not found. Falling back to slow Python loop (expected on Mac).")

# ==========================================
# SHARED UTILITIES & FFN
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SwiGLU_FFN(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier=4):
        super().__init__()
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# ==========================================
# 1. INDUSTRY STANDARD LLM
# ==========================================
class StandardTransformerLayer(nn.Module):
    def __init__(self, dim, nhead):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffwd = SwiGLU_FFN(dim)

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ffwd(self.norm2(x))
        return x

class IndustryStandardLLM(nn.Module):
    def __init__(self, vocab_size, num_classes, dim, nhead, num_layers, max_seq_len=1000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_encoder = PositionalEncoding(dim, max_seq_len)
        self.layers = nn.ModuleList([StandardTransformerLayer(dim, nhead) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x, return_residual=False):
        h = self.embed(x)
        h = self.pos_encoder(h) 
        
        for layer in self.layers:
            h = layer(h)
            
        residual_state = h 
        h_final = self.layer_norm(h[:, -1, :])
        logits = self.fc(h_final)
        
        if return_residual:
            return logits, residual_state
        return logits

# ==========================================
# 2. RECURRENT Q/K (MAMBA/HIPPO) TRANSFORMER
# ==========================================
class RecurrentAttention(nn.Module):
    def __init__(self, dim, num_heads, state_dim=16):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.state_dim = state_dim

        # Forward Scan Parameters
        A_fwd = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(dim, 1)
        self.A_log_fwd = nn.Parameter(torch.log(A_fwd))
        self.proj_delta_fwd = nn.Linear(dim, dim)
        self.proj_B_fwd = nn.Linear(dim, state_dim, bias=False)
        self.proj_C_fwd = nn.Linear(dim, state_dim, bias=False)

        # Backward Scan Parameters
        A_bwd = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(dim, 1)
        self.A_log_bwd = nn.Parameter(torch.log(A_bwd))
        self.proj_delta_bwd = nn.Linear(dim, dim)
        self.proj_B_bwd = nn.Linear(dim, state_dim, bias=False)
        self.proj_C_bwd = nn.Linear(dim, state_dim, bias=False)

        # Base projections
        self.x_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def _selective_scan(self, x_base, x_raw, proj_delta, proj_B, proj_C, A_log):
        B_batch, T, C_dim = x_raw.shape
        device = x_raw.device

        delta = proj_delta(x_raw)
        B_mat = proj_B(x_raw)
        C_mat = proj_C(x_raw)
        A = -torch.exp(A_log).to(device)

        if HAS_MAMBA and device.type == 'cuda':
            x_base_t = x_base.transpose(1, 2).contiguous()
            delta_t = delta.transpose(1, 2).contiguous()
            B_mat_t = B_mat.transpose(1, 2).contiguous()
            C_mat_t = C_mat.transpose(1, 2).contiguous()

            out_t = selective_scan_fn(
                x_base_t, delta_t, A, B_mat_t, C_mat_t, 
                None, None, None, delta_softplus=True
            )
            return out_t.transpose(1, 2).contiguous()

        h = torch.zeros(B_batch, C_dim, self.state_dim, device=device)
        out_seq = []

        for t in range(T):
            x_t = x_base[:, t, :].unsqueeze(-1)
            delta_t = delta[:, t, :].unsqueeze(-1)
            B_t = B_mat[:, t, :].unsqueeze(1)
            C_t = C_mat[:, t, :].unsqueeze(1)

            bar_A = torch.exp(delta_t * A)
            bar_B = delta_t * B_t

            h = bar_A * h + bar_B * x_t
            y_t = (h * C_t).sum(dim=-1)
            out_seq.append(y_t)

        return torch.stack(out_seq, dim=1)

    def forward(self, x):
        B_batch, T, C_dim = x.shape

        x_base = self.x_proj(x)

        # 1. Forward Scan (Left-to-Right)
        out_fwd = self._selective_scan(
            x_base, x, 
            self.proj_delta_fwd, self.proj_B_fwd, self.proj_C_fwd, self.A_log_fwd
        )

        # 2. Backward Scan (Right-to-Left)
        # Flip the sequence along the time dimension (dim=1)
        x_base_flipped = torch.flip(x_base, dims=[1])
        x_flipped = torch.flip(x, dims=[1])
        
        out_bwd_flipped = self._selective_scan(
            x_base_flipped, x_flipped, 
            self.proj_delta_bwd, self.proj_B_bwd, self.proj_C_bwd, self.A_log_bwd
        )
        
        # Flip the backward output back to match the original sequence order
        out_bwd = torch.flip(out_bwd_flipped, dims=[1])

        # 3. Combine Bidirectional Context for Q and K
        # Adding the forward and backward representations (can also be concatenated)
        combined_state = out_fwd + out_bwd

        Q = combined_state.view(B_batch, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = combined_state.view(B_batch, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B_batch, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)

        # 4. Standard Attention Execution
        scores = (Q @ K.transpose(-2, -1)) * F.softplus(self.temperature)
        attn = scores.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B_batch, T, C_dim)

        return self.proj_out(out)

class RecurrentTransformerLayer(nn.Module):
    def __init__(self, dim, nhead):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RecurrentAttention(dim, nhead)
        self.norm2 = nn.LayerNorm(dim)
        self.ffwd = SwiGLU_FFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffwd(self.norm2(x))
        return x

class RecurrentLLM(nn.Module):
    def __init__(self, vocab_size, num_classes, dim, nhead, num_layers, max_seq_len=1000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        # Note: Recurrent/HiPPO models often don't strictly need positional encoding, 
        # but we keep it here for an apples-to-apples comparison.
        self.pos_encoder = PositionalEncoding(dim, max_seq_len)
        self.layers = nn.ModuleList([RecurrentTransformerLayer(dim, nhead) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x, return_residual=False):
        h = self.embed(x)
        h = self.pos_encoder(h)
        for layer in self.layers:
            h = layer(h)
        residual_state = h 
        h_final = self.layer_norm(h[:, -1, :])
        logits = self.fc(h_final)
        if return_residual:
            return logits, residual_state
        return logits
    
class UniversalACTWrapper(nn.Module):
    def __init__(self, layer, dim, max_steps=12):
        super().__init__()
        self.layer = layer
        self.max_steps = max_steps
        
        self.step_embed = nn.Embedding(max_steps + 1, dim)
        self.halting_gate = nn.Linear(dim, 1)
        self.halting_gate.bias.data.fill_(-8.0) # Strong bias to encourage depth

    def forward(self, x):
        B, T, C = x.shape
        device = x.device

        # Track probabilities globally for the sequence (B, 1, 1)
        accumulated_probs = torch.zeros(B, 1, 1, device=device)
        updates = torch.zeros(B, 1, 1, device=device)
        active_mask = torch.ones(B, 1, 1, device=device, dtype=torch.bool)
        
        output_state = torch.zeros_like(x)
        
        for step in range(1, self.max_steps + 1):
            step_tensor = torch.full((B, T), step, device=device, dtype=torch.long)
            step_emb = self.step_embed(step_tensor)
            
            x_in = x + step_emb
            x = self.layer(x_in)
            
            # Predict halting per token, then average across sequence
            raw_p_t = torch.sigmoid(self.halting_gate(x)) 
            p_t = raw_p_t.mean(dim=1, keepdim=True)       
            
            if step == self.max_steps:
                is_halting_now = active_mask
            else:
                is_halting_now = (accumulated_probs + p_t >= 1.0) & active_mask
            
            step_weight = torch.where(
                is_halting_now, 
                1.0 - accumulated_probs, 
                p_t
            )
            step_weight = step_weight * active_mask.float()
            
            output_state = output_state + (step_weight * x)
            
            accumulated_probs = accumulated_probs + step_weight
            updates = updates + active_mask.float()
            active_mask = active_mask & ~is_halting_now
            
            if not active_mask.any():
                break
                
        ponder_cost = updates.mean()
        return output_state, ponder_cost

class UniversalLLM(nn.Module):
    def __init__(self, vocab_size, num_classes, dim, nhead, max_steps=12, max_seq_len=5000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_encoder = PositionalEncoding(dim, max_seq_len) 
        
        core_layer = RecurrentTransformerLayer(dim, nhead)
        self.universal_layer = UniversalACTWrapper(core_layer, dim, max_steps=max_steps)
        
        self.layer_norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x, return_residual=False):
        h = self.embed(x)
        h = self.pos_encoder(h)
        
        h, ponder_cost = self.universal_layer(h)
        residual_state = h 
        
        # Pool across sequence for final classification
        h_pooled = h.mean(dim=1)
        h_final = self.layer_norm(h_pooled)
        
        logits = self.fc(h_final)
        
        if return_residual:
            return logits, ponder_cost, residual_state
        return logits, ponder_cost