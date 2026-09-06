"""Utilities for the Humanoid decomposition/alignment experiment.

The experiment deliberately uses the same MLP family for every actor and the
critic.  This makes the parameter-matching calculation exact instead of relying
on an approximate count from a different model implementation.
"""

VALID_ALIGNMENT_MODES = {
    "separate",
    "hard_share",
    "critic_to_actor",
    "actor_to_critic",
    "bidirectional",
    "no_stop",
}

ACTOR_ALIGNMENT_MODES = {"critic_to_actor", "bidirectional"}
CRITIC_ALIGNMENT_MODES = {"actor_to_critic", "bidirectional"}


def mlp_base_parameter_count(input_dim, width, depth, feature_normalization=True):
    """Return the exact parameter count of ``MLPBase`` for uniform widths."""
    if input_dim <= 0 or width <= 0 or depth <= 0:
        raise ValueError("input_dim, width, and depth must be positive")

    total = 2 * input_dim if feature_normalization else 0
    previous = input_dim
    for _ in range(depth):
        # Linear weight + bias, followed by LayerNorm weight + bias.
        total += previous * width + width + 2 * width
        previous = width
    return total


def experiment_parameter_count(
    input_dim,
    action_dims,
    width,
    depth,
    hard_share=False,
    feature_normalization=True,
):
    """Count unique actor/critic parameters for the experiment architecture.

    Continuous actor heads contain a mean projection and a learned log standard
    deviation.  The centralized value head is one linear layer.
    """
    if not action_dims or any(dim <= 0 for dim in action_dims):
        raise ValueError("action_dims must contain positive dimensions")

    base_count = mlp_base_parameter_count(
        input_dim, width, depth, feature_normalization
    )
    base_copies = 1 if hard_share else len(action_dims) + 1
    actor_heads = sum(width * dim + 2 * dim for dim in action_dims)
    critic_head = width + 1
    return base_copies * base_count + actor_heads + critic_head


def resolve_parameter_matched_width(
    input_dim,
    action_dims,
    target_parameters,
    depth,
    hard_share=False,
    feature_normalization=True,
    min_width=8,
    max_width=2048,
):
    """Choose the integer width whose exact count is closest to the target."""
    if target_parameters <= 0:
        raise ValueError("target_parameters must be positive")
    if min_width > max_width:
        raise ValueError("min_width must not exceed max_width")

    best = None
    for width in range(min_width, max_width + 1):
        count = experiment_parameter_count(
            input_dim,
            action_dims,
            width,
            depth,
            hard_share=hard_share,
            feature_normalization=feature_normalization,
        )
        candidate = (abs(count - target_parameters), width, count)
        if best is None or candidate < best:
            best = candidate
    _, width, count = best
    return width, count


def unique_trainable_parameters(modules):
    """Return trainable parameters once, even when modules share an encoder."""
    parameters = []
    seen = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    return parameters


def count_unique_trainable_parameters(modules):
    """Count unique trainable scalar parameters across modules."""
    return sum(parameter.numel() for parameter in unique_trainable_parameters(modules))


def normalized_alignment_loss(source, target, active_masks=None, normalize=True):
    """MSE alignment loss with optional parameter-free feature normalization."""
    import torch.nn.functional as functional

    if source.shape != target.shape:
        raise ValueError(
            f"alignment shape mismatch: source={source.shape}, target={target.shape}"
        )
    if normalize:
        source = functional.layer_norm(source, [source.shape[-1]])
        target = functional.layer_norm(target, [target.shape[-1]])

    loss = functional.mse_loss(source, target, reduction="none").mean(
        dim=-1, keepdim=True
    )
    if active_masks is None:
        return loss.mean()
    denominator = active_masks.sum().clamp_min(1.0)
    return (loss * active_masks).sum() / denominator
