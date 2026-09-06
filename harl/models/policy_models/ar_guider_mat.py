# harl/models/policy_models/ar_guider_mat.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.utils.envs_tools import check, get_shape_from_obs_space, get_shape_from_act_space
from harl.models.base.act import ACTLayer

def init_linear(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

class RMSNorm(nn.Module):
    """Matches Flax's nn.RMSNorm used in Mava's MAT if configured."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return (x / (norm + self.eps)) * self.weight

class SwiGLU(nn.Module):
    """Exact match for Mava's SwiGLU used in MAT"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w1 = nn.Linear(in_dim, out_dim * 2, bias=False)
        self.w2 = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, x):
        x1, x2 = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(x1) * x2)

class ARGuiderPolicyMAT(nn.Module):
    """
    Autoregressive Guider Policy strictly aligned with Mava's Multi-Agent Transformer (MAT).
    Uses standard Encoder-Decoder Attention mechanisms.
    """
    def __init__(self, args, share_obs_space, action_space, num_agents, device=torch.device("cpu")):
        super(ARGuiderPolicyMAT, self).__init__()
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = num_agents
        self.action_space_type = action_space.__class__.__name__
        
        self.embed_dim = args.get("hidden_sizes", [64, 64])[-1]
        n_head = args.get("transformer_heads", 2)
        n_block = args.get("transformer_blocks", 2)
        use_rmsnorm = args.get("use_rmsnorm", False) # MAT in Mava can use RMSNorm or LayerNorm
        
        NormLayer = RMSNorm if use_rmsnorm else nn.LayerNorm
        
        # --- 1. Observation Encoder (Encoder in MAT) ---
        share_obs_shape = get_shape_from_obs_space(share_obs_space)[0]
        self.obs_encoder = nn.Sequential(
            NormLayer(share_obs_shape),
            init_linear(nn.Linear(share_obs_shape, self.embed_dim, bias=False), nn.init.orthogonal_, nn.init.zeros_, gain=1.414),
            nn.GELU()
        )
        
        # --- 2. Action Encoder (Decoder in MAT) ---
        act_shape = get_shape_from_act_space(action_space)
        if self.action_space_type == "Discrete":
            self.action_dim = action_space.n
            self.action_encoder = nn.Sequential(
                nn.Embedding(self.action_dim, self.embed_dim),
                nn.GELU()
            )
        else:
            self.action_dim = act_shape if isinstance(act_shape, int) else act_shape[0]
            self.action_encoder = nn.Sequential(
                init_linear(nn.Linear(self.action_dim, self.embed_dim, bias=False), nn.init.orthogonal_, nn.init.zeros_, gain=1.414),
                nn.GELU()
            )
            
        self.start_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        
        # --- 3. Transformer Engine (Self-Attention + Cross-Attention) ---
        # Note: We use PyTorch's builtin Transformer, which matches MAT's EncodeBlock and DecodeBlock.
        # The cross-attention logic in Mava MAT is standard (Query=Decoder, Key/Value=Encoder).
        self.transformer = nn.Transformer(
            d_model=self.embed_dim,
            nhead=n_head,
            num_encoder_layers=n_block,
            num_decoder_layers=n_block,
            dim_feedforward=self.embed_dim * 2,
            batch_first=True,
            activation='gelu',
            norm_first=True  # Standard Pre-Norm formulation
        )
        
        # --- 4. Action Head (ACTLayer) ---
        self.dec_ln = NormLayer(self.embed_dim)
        self.act_head = ACTLayer(action_space, self.embed_dim, args["initialization_method"], args["gain"], args)
        
        self.to(device)

    def _generate_causal_mask(self, sz):
        """Generates upper-triangular mask for autoregressive decoding."""
        mask = (torch.triu(torch.ones(sz, sz, device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _encode_obs(self, share_obs):
        """MAT uses the global state broadcasted to N tokens (agents_view)."""
        if len(share_obs.shape) == 3:
            obs_seq = share_obs
        elif len(share_obs.shape) == 2:
            obs_seq = share_obs.unsqueeze(1).expand(-1, self.num_agents, -1)
        else:
            raise ValueError(f"Unexpected share_obs shape: {share_obs.shape}")

        obs_embs = self.obs_encoder(obs_seq)
        # Pass through Transformer Encoder
        memory = self.transformer.encoder(obs_embs)
        return memory

    def get_actions(self, share_obs, available_actions=None, deterministic=False):
        """Rollout Phase: Autoregressive decoding (O(N) Loop)."""
        share_obs = check(share_obs).to(**self.tpdv)
        batch_size = share_obs.shape[0]
        
        # [Batch, N, Dim]
        memory = self._encode_obs(share_obs) 
        
        actions = []
        action_log_probs = []
        
        # Initialize with Start Token
        tgt_seq = self.start_token.expand(batch_size, 1, -1)
        
        for agent_id in range(self.num_agents):
            # Causal mask for Decoder
            tgt_mask = self._generate_causal_mask(agent_id + 1)
            
            # Forward pass through Decoder
            decoder_out = self.transformer.decoder(tgt_seq, memory, tgt_mask=tgt_mask)
            
            # Take the feature of the current agent's step
            step_feat = self.dec_ln(decoder_out[:, -1, :]) 
            
            avail_act = available_actions[:, agent_id] if available_actions is not None else None
            action, action_log_prob = self.act_head(step_feat, avail_act, deterministic)
            
            actions.append(action)
            action_log_probs.append(action_log_prob)
            
            # Encode action and append to tgt_seq for next step
            if self.action_space_type == "Discrete":
                act_emb = self.action_encoder(action.squeeze(-1).long())
            else:
                act_emb = self.action_encoder(action).unsqueeze(1)
                
            tgt_seq = torch.cat([tgt_seq, act_emb], dim=1)
            
        return torch.stack(actions, dim=1), torch.stack(action_log_probs, dim=1)

    def evaluate_actions(self, share_obs, joint_actions, available_actions=None, active_masks=None):
        """Training Phase: Chunkwise parallel execution using Teacher Forcing."""
        share_obs = check(share_obs).to(**self.tpdv)
        joint_actions = check(joint_actions).to(**self.tpdv)
        batch_size = share_obs.shape[0]
        
        # [Batch, N, Dim]
        memory = self._encode_obs(share_obs)
        
        if self.action_space_type == "Discrete":
            act_embs = self.action_encoder(joint_actions.squeeze(-1).long())
        else:
            act_embs = self.action_encoder(joint_actions)
            
        # Teacher Forcing: Shift right by 1
        start_tokens = self.start_token.expand(batch_size, 1, -1)
        tgt_seq = torch.cat([start_tokens, act_embs[:, :-1, :]], dim=1)
        
        tgt_mask = self._generate_causal_mask(self.num_agents)
        
        # Parallel Decode
        decoder_out = self.transformer.decoder(tgt_seq, memory, tgt_mask=tgt_mask)
        decoder_out = self.dec_ln(decoder_out)
        
        out_feat_flat = decoder_out.reshape(-1, self.embed_dim)
        joint_actions_flat = joint_actions.reshape(-1, joint_actions.shape[-1])
        avail_act_flat = available_actions.reshape(-1, available_actions.shape[-1]) if available_actions is not None else None
        active_masks_flat = active_masks.reshape(-1, 1) if active_masks is not None else None
            
        action_log_probs_flat, dist_entropy, action_distribution = self.act_head.evaluate_actions(
            out_feat_flat, joint_actions_flat, avail_act_flat, active_masks_flat
        )
        return action_log_probs_flat.reshape(batch_size, self.num_agents, -1), dist_entropy, action_distribution