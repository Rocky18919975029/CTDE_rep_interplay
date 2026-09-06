# harl/models/policy_models/ar_guider_sable.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.utils.envs_tools import check, get_shape_from_obs_space, get_shape_from_act_space
from harl.models.base.act import ACTLayer

class RMSNorm(nn.Module):
    """Exact match for Flax's nn.RMSNorm"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return (x / (norm + self.eps)) * self.weight

class SwiGLU(nn.Module):
    """Exact match for Mava's SwiGLU"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w1 = nn.Linear(in_dim, out_dim * 2, bias=False)
        self.w2 = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, x):
        x1, x2 = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(x1) * x2)

class SimpleRetention(nn.Module):
    """Translated from mava.networks.retention.SimpleRetention"""
    def __init__(self, embed_dim, head_size, masked, decay_kappa):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_size = head_size
        self.masked = masked
        self.decay_kappa = decay_kappa

        self.w_q = nn.Linear(embed_dim, head_size, bias=False)
        self.w_k = nn.Linear(embed_dim, head_size, bias=False)
        self.w_v = nn.Linear(embed_dim, head_size, bias=False)
        
        # Mava initializes with normal stddev=1/embed_dim
        nn.init.normal_(self.w_q.weight, std=1.0 / embed_dim)
        nn.init.normal_(self.w_k.weight, std=1.0 / embed_dim)
        nn.init.normal_(self.w_v.weight, std=1.0 / embed_dim)

    def _get_decay_matrix(self, seq_len, device):
        n = torch.arange(seq_len, device=device).unsqueeze(1)
        m = torch.arange(seq_len, device=device).unsqueeze(0)
        decay_matrix = (self.decay_kappa ** (n - m)) * (n >= m)
        if self.masked:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            decay_matrix = decay_matrix * mask
        return decay_matrix.unsqueeze(0) # [1, S, S]

    def forward(self, key, query, value):
        """Chunkwise representation (used in training)"""
        B, S, _ = value.shape
        q_proj = self.w_q(query) # [B, S, H]
        k_proj = self.w_k(key).transpose(1, 2) # [B, H, S]
        v_proj = self.w_v(value) # [B, S, H]

        decay_matrix = self._get_decay_matrix(S, value.device)
        
        # inner_chunk = ((q_proj @ k_proj) * decay_matrix) @ v_proj
        scores = torch.bmm(q_proj, k_proj) * decay_matrix
        ret = torch.bmm(scores, v_proj)
        return ret

    def recurrent(self, key_n, query_n, value_n, hstate):
        """Recurrent representation (used in rollout/inference)"""
        q_proj = self.w_q(query_n) # [B, 1, H]
        k_proj = self.w_k(key_n).transpose(1, 2) # [B, H, 1]
        v_proj = self.w_v(value_n) # [B, 1, H]

        updated_hstate = hstate + torch.bmm(k_proj, v_proj) # [B, H, H]
        ret = torch.bmm(q_proj, updated_hstate) # [B, 1, H]
        
        return ret, updated_hstate

class MultiScaleRetention(nn.Module):
    """Translated from mava.networks.retention.MultiScaleRetention"""
    def __init__(self, embed_dim, n_head, masked):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_size = embed_dim // n_head
        self.masked = masked

        # Exact decay_kappas calculation from JAX source
        decay_logs = torch.linspace(math.log(1/32), math.log(1/512), n_head)
        self.decay_kappas = 1 - torch.exp(decay_logs) # List/Tensor of kappas
        
        self.w_g = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_o = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.normal_(self.w_g.weight, std=1.0 / embed_dim)
        nn.init.normal_(self.w_o.weight, std=1.0 / embed_dim)

        self.group_norm = nn.GroupNorm(num_groups=n_head, num_channels=embed_dim)

        self.heads = nn.ModuleList([
            SimpleRetention(embed_dim, self.head_size, masked, dk.item())
            for dk in self.decay_kappas
        ])

    def forward(self, key, query, value):
        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(key, query, value))
            
        ret_output = torch.cat(head_outputs, dim=-1) # [B, S, E]
        
        # GroupNorm expects [B, C, S]
        ret_output = ret_output.transpose(1, 2)
        ret_output = self.group_norm(ret_output).transpose(1, 2)
        
        output = self.w_o(F.silu(self.w_g(key)) * ret_output)
        return output

    def recurrent(self, key_n, query_n, value_n, hstate):
        head_outputs = []
        new_hstates = []
        
        for i, head in enumerate(self.heads):
            ret, hs_new = head.recurrent(key_n, query_n, value_n, hstate[:, i])
            head_outputs.append(ret)
            new_hstates.append(hs_new)
            
        ret_output = torch.cat(head_outputs, dim=-1)
        ret_output = ret_output.transpose(1, 2)
        ret_output = self.group_norm(ret_output).transpose(1, 2)
        
        output = self.w_o(F.silu(self.w_g(key_n)) * ret_output)
        return output, torch.stack(new_hstates, dim=1)

class SableEncodeBlock(nn.Module):
    def __init__(self, embed_dim, n_head):
        super().__init__()
        self.ln1 = RMSNorm(embed_dim)
        self.ln2 = RMSNorm(embed_dim)
        self.retn = MultiScaleRetention(embed_dim, n_head, masked=False)
        self.ffn = SwiGLU(embed_dim, embed_dim)

    def forward(self, x):
        ret = self.retn(key=x, query=x, value=x)
        x = self.ln1(x + ret)
        return self.ln2(x + self.ffn(x))

class SableDecodeBlock(nn.Module):
    def __init__(self, embed_dim, n_head):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = RMSNorm(embed_dim), RMSNorm(embed_dim), RMSNorm(embed_dim)
        self.retn1 = MultiScaleRetention(embed_dim, n_head, masked=True) # Self retention
        self.retn2 = MultiScaleRetention(embed_dim, n_head, masked=True) # Cross retention
        self.ffn = SwiGLU(embed_dim, embed_dim)

    def forward(self, x, obs_rep):
        ret = self.retn1(key=x, query=x, value=x)
        ret = self.ln1(x + ret)
        
        # Mava Special Cross-Retention: key/value are 'ret', query is 'obs_rep'
        ret2 = self.retn2(key=ret, query=obs_rep, value=ret)
        y = self.ln2(obs_rep + ret2)
        
        return self.ln3(y + self.ffn(y))

    def recurrent(self, x, obs_rep, hstates):
        hs1, hs2 = hstates
        ret, hs1_new = self.retn1.recurrent(key_n=x, query_n=x, value_n=x, hstate=hs1)
        ret = self.ln1(x + ret)
        
        ret2, hs2_new = self.retn2.recurrent(key_n=ret, query_n=obs_rep, value_n=ret, hstate=hs2)
        y = self.ln2(obs_rep + ret2)
        
        return self.ln3(y + self.ffn(y)), (hs1_new, hs2_new)

class ARGuiderPolicySable(nn.Module):
    """
    Autoregressive Guider Policy. 
    Strictly aligned with mava/networks/sable_network.py (SableNetwork).
    """
    def __init__(self, args, share_obs_space, action_space, num_agents, device=torch.device("cpu")):
        super(ARGuiderPolicySable, self).__init__()
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = num_agents
        self.action_space_type = action_space.__class__.__name__
        
        self.embed_dim = args.get("hidden_sizes", [64, 64])[-1]
        self.n_head = args.get("transformer_heads", 2)
        self.n_block = args.get("transformer_blocks", 2)
        self.head_size = self.embed_dim // self.n_head

        # Precompute decay kappas for hstate decay
        decay_logs = torch.linspace(math.log(1/32), math.log(1/512), self.n_head)
        self.decay_kappas = (1 - torch.exp(decay_logs)).view(1, self.n_head, 1, 1).to(device)

        # --- Encoder ---
        share_obs_shape = get_shape_from_obs_space(share_obs_space)[0]
        self.obs_encoder = nn.Sequential(
            RMSNorm(share_obs_shape),
            nn.Linear(share_obs_shape, self.embed_dim, bias=False),
            nn.GELU()
        )
        self.enc_ln = RMSNorm(self.embed_dim)
        self.enc_blocks = nn.ModuleList([SableEncodeBlock(self.embed_dim, self.n_head) for _ in range(self.n_block)])

        # --- Decoder ---
        act_shape = get_shape_from_act_space(action_space)
        if self.action_space_type == "Discrete":
            self.action_dim = action_space.n
            self.action_encoder = nn.Sequential(nn.Embedding(self.action_dim, self.embed_dim), nn.GELU())
        else:
            self.action_dim = act_shape if isinstance(act_shape, int) else act_shape[0]
            self.action_encoder = nn.Sequential(nn.Linear(self.action_dim, self.embed_dim, bias=False), nn.GELU())
            
        self.dec_ln = RMSNorm(self.embed_dim)
        self.dec_blocks = nn.ModuleList([SableDecodeBlock(self.embed_dim, self.n_head) for _ in range(self.n_block)])

        self.start_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.act_head = ACTLayer(action_space, self.embed_dim, args["initialization_method"], args["gain"], args)
        
        self.to(device)

    def _encode_obs(self, share_obs):
        # 如果传入的已经是 [Batch, N, Dim]，说明环境本身是按 agent 分配全局状态的
        if len(share_obs.shape) == 3:
            obs_seq = share_obs
        # 如果传入的是 [Batch, Dim] 的单一全局向量，我们将其编码后广播给所有 N 个 Agent
        elif len(share_obs.shape) == 2:
            # 增加一个维度，利用广播机制，[Batch, Dim] -> [Batch, 1, Dim]
            obs_seq = share_obs.unsqueeze(1).expand(-1, self.num_agents, -1)
        else:
            raise ValueError(f"Unexpected share_obs shape: {share_obs.shape}")

        x = self.obs_encoder(obs_seq)
        x = self.enc_ln(x)
        for block in self.enc_blocks:
            x = block(x)
        return x

    def get_actions(self, share_obs, available_actions=None, deterministic=False):
        """Rollout Phase: Autoregressive decoding using recurrent state."""
        share_obs = check(share_obs).to(**self.tpdv)
        batch_size = share_obs.shape[0]
        
        obs_rep = self._encode_obs(share_obs) # [B, N, D]
        
        # Init Recurrent Hidden States: [B, n_head, head_size, head_size]
        hstates = [[torch.zeros(batch_size, self.n_head, self.head_size, self.head_size, device=self.device) for _ in range(2)] for _ in range(self.n_block)]
        
        actions, action_log_probs = [], []
        x = self.start_token.expand(batch_size, 1, -1)
        
        for agent_id in range(self.num_agents):
            x_in = self.dec_ln(x)
            
            # Decay hidden states
            for i in range(self.n_block):
                hstates[i][0] = hstates[i][0] * self.decay_kappas
                hstates[i][1] = hstates[i][1] * self.decay_kappas
                
            # Forward Decode Blocks
            for i, block in enumerate(self.dec_blocks):
                x_in, (hs1_new, hs2_new) = block.recurrent(x_in, obs_rep[:, agent_id:agent_id+1, :], hstates[i])
                hstates[i][0], hstates[i][1] = hs1_new, hs2_new
                
            avail_act = available_actions[:, agent_id] if available_actions is not None else None
            action, action_log_prob = self.act_head(x_in.squeeze(1), avail_act, deterministic)
            
            actions.append(action)
            action_log_probs.append(action_log_prob)
            
            if self.action_space_type == "Discrete":
                x = self.action_encoder(action.squeeze(-1).long()).unsqueeze(1)
            else:
                x = self.action_encoder(action).unsqueeze(1)
                
        return torch.stack(actions, dim=1), torch.stack(action_log_probs, dim=1)

    def evaluate_actions(self, share_obs, joint_actions, available_actions=None, active_masks=None):
        """Training Phase: Chunkwise parallel execution using Teacher Forcing."""
        share_obs = check(share_obs).to(**self.tpdv)
        joint_actions = check(joint_actions).to(**self.tpdv)
        batch_size = share_obs.shape[0]
        
        obs_rep = self._encode_obs(share_obs)
        
        if self.action_space_type == "Discrete":
            act_embs = self.action_encoder(joint_actions.squeeze(-1).long())
        else:
            act_embs = self.action_encoder(joint_actions)
            
        start_tokens = self.start_token.expand(batch_size, 1, -1)
        tgt_seq = torch.cat([start_tokens, act_embs[:, :-1, :]], dim=1)
        
        x = self.dec_ln(tgt_seq)
        for block in self.dec_blocks:
            x = block(x, obs_rep)
            
        out_feat_flat = x.reshape(-1, self.embed_dim)
        joint_actions_flat = joint_actions.reshape(-1, joint_actions.shape[-1])
        avail_act_flat = available_actions.reshape(-1, available_actions.shape[-1]) if available_actions is not None else None
        active_masks_flat = active_masks.reshape(-1, 1) if active_masks is not None else None
            
        action_log_probs_flat, dist_entropy, action_distribution = self.act_head.evaluate_actions(
            out_feat_flat, joint_actions_flat, avail_act_flat, active_masks_flat
        )
        return action_log_probs_flat.reshape(batch_size, self.num_agents, -1), dist_entropy, action_distribution