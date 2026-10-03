# Copyright (c) ModelScope Contributors. All rights reserved.
"""Megatron MoE routing-replay utilities.

The Megatron counterpart of ``twinkle.model.transformers.moe.router_replay``. Both backends expose the
same public surface (``RouterReplayAction`` + ``set_global_router_replay_action`` /
``clear_global_router_replay_action`` / ``clear_global_indices`` / ``set_router_replay_data`` /
``get_router_replay_data`` / ``apply_router_replay_patch``) so ``MegatronModel`` can drive routing
replay symmetrically to ``TransformersModel`` -- basic principle 1 (the two training backends are
equivalent except for ``generate``).

Unlike the FSDP/transformers path, which owns its own per-block replay registry, Megatron delegates the
record/replay state to mcore's ``megatron.core.transformer.moe.router_replay.RouterReplay`` (one instance per
MoE router, kept in ``RouterReplay.global_router_replay_instances``). What lives here is the Megatron-only
glue mcore does not provide: mapping a whole-model ``routed_experts`` tensor (as reported across ALL
transformer layers, including dense ones) onto the LOCAL (pp, vp, cp, sp) layer/token slice each rank
actually routes, and the ``MoEAlltoAllTokenDispatcher`` patch that keeps the all-to-all split sizes
consistent when replayed indices contain duplicates.

Requires ``megatron-core >= 0.16`` (the version that ships ``RouterReplay``); when it is absent every
entry point raises rather than silently degrading to recomputed routing.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from twinkle.utils import get_logger

logger = get_logger()

try:
    from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    ROUTER_REPLAY_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the installed megatron-core version
    logger.warning('RouterReplay not available in current megatron-core version')
    RouterReplay = None
    RouterReplayAction = None
    MoEAlltoAllTokenDispatcher = None
    ROUTER_REPLAY_AVAILABLE = False


def _require_available() -> None:
    """Fail loudly when mcore lacks ``RouterReplay`` instead of silently recomputing routing.

    Recomputed routing makes an MoE policy's importance ratio silently wrong, so an unavailable
    ``megatron-core`` must abort rather than degrade.
    """
    if not ROUTER_REPLAY_AVAILABLE:
        raise RuntimeError('Megatron routing replay is not supported: upgrade megatron-core to 0.16.0 '
                           'or higher (mcore.transformer.moe.router_replay.RouterReplay is missing).')


def _tf_config(model: Any) -> Any:
    """Return the mcore ``TransformerConfig`` of an (unwrapped) Megatron model chunk."""
    config = getattr(model, 'config', None)
    if config is None:
        raise ValueError('Megatron routing replay needs a model exposing `.config` (the mcore '
                         f'TransformerConfig); got {type(model).__name__} without one.')
    return config


def resolve_router_replay_action(action: Any) -> Any:
    """Resolve a backend-neutral action token to this backend's ``RouterReplayAction`` enum.

    Callers (swift/dev) stay backend-agnostic per basic principle 1, so they pass the action as a plain
    string (``'record'`` / ``'replay_forward'`` / ``'replay_backward'``) rather than importing a
    backend-specific enum. ``None`` passes through (replay disabled for this forward); an already-enum
    value passes through; a string is looked up by value.
    """
    _require_available()
    if action is None or isinstance(action, RouterReplayAction):
        return action
    if isinstance(action, str):
        try:
            return RouterReplayAction(action)
        except ValueError:
            raise ValueError(f'Unknown router_replay_action {action!r}; expected one of '
                             f'{[a.value for a in RouterReplayAction]} or None.') from None
    raise TypeError(f'router_replay_action must be a str, RouterReplayAction or None, got {type(action).__name__}.')



# ---------------------------------------------------------------------------
# Layer / rank slicing (migrated verbatim from legacy swift's router_replay_utils)
# ---------------------------------------------------------------------------


def is_moe_layer(tf_config: Any, layer_idx: int) -> bool:
    moe_layer_freq = getattr(tf_config, 'moe_layer_freq', None)
    if isinstance(moe_layer_freq, int):
        return layer_idx % moe_layer_freq == 0
    elif isinstance(moe_layer_freq, list):
        return moe_layer_freq[layer_idx] == 1
    else:
        raise ValueError(f'Unsupported moe_layer_freq type: {type(moe_layer_freq)}')


def get_moe_num_layers_to_build(tf_config: Any, vp_stage: Optional[int] = None,
                                pp_rank: Optional[int] = None) -> int:
    """Count the number of MoE layers assigned to the current rank.

    When ``moe_layer_freq`` is 1 or unset, every transformer layer is an MoE layer, so the count equals
    the total layer count. Otherwise only layers whose global index satisfies the frequency predicate
    are counted.
    """
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    total_layers = get_num_layers_to_build(tf_config, vp_stage=vp_stage, pp_rank=pp_rank)
    layer_offset = get_transformer_layer_offset(tf_config, vp_stage=vp_stage)
    local_global_indices = range(layer_offset, layer_offset + total_layers)
    return sum(1 for idx in local_global_indices if is_moe_layer(tf_config, idx))


def get_local_layer_range(tf_config: Any, vp_rank: Optional[int] = None, only_moe_layer: bool = True):
    """Return ``(offset, count)`` -- the local router-instance range within the global instance list."""
    from megatron.core.transformer.transformer_block import get_num_layers_to_build

    vp_size = tf_config.virtual_pipeline_model_parallel_size
    if vp_size is not None:
        vp_rank = 0 if vp_rank is None else vp_rank
        offset = 0
        for pre_vp_stage in range(vp_size):
            if pre_vp_stage == vp_rank:
                break
            num_layers_to_build = get_moe_num_layers_to_build(
                tf_config, pre_vp_stage) if only_moe_layer else get_num_layers_to_build(tf_config, pre_vp_stage)
            offset += num_layers_to_build
    else:
        offset = 0
    count = get_moe_num_layers_to_build(tf_config, vp_rank) if only_moe_layer else get_num_layers_to_build(
        tf_config, vp_rank)
    return offset, count


def get_local_topk_idx_for_current_rank(global_topk_idx: Optional[torch.Tensor],
                                        tf_config: Any,
                                        packed_seq_params: Any = None) -> Optional[torch.Tensor]:
    """Slice a whole-model ``routed_experts`` tensor down to this rank's local (pp, cp, sp) shard.

    ``global_topk_idx`` is ``[bs, seq_len, layers, topk]`` reported across ALL transformer layers
    (vLLM emits routing for dense layers too), but Megatron only has routers for the MoE layers, so the
    layer dimension is filtered to this rank's local MoE layers, then the sequence dimension is sliced for
    context parallel (cp) and sequence parallel (sp).
    """
    from megatron.core import mpu
    from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    if global_topk_idx is None:
        return None
    # 1. pp slice: keep only this rank's MoE layers out of the all-layer routing report.
    layer_offset = get_transformer_layer_offset(tf_config, vp_stage=0)
    offset, count = get_local_layer_range(tf_config, tf_config.virtual_pipeline_model_parallel_size,
                                          only_moe_layer=False)
    num_layers = offset + count
    moe_layer_idx = torch.tensor(
        [layer_idx for layer_idx in range(layer_offset, layer_offset + num_layers)
         if is_moe_layer(tf_config, layer_idx)],
        dtype=torch.long,
        device=global_topk_idx.device)
    local_topk_idx = torch.index_select(global_topk_idx, dim=2, index=moe_layer_idx)
    # 2. cp slice.
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        from mcore_bridge import split_cp_inputs
        local_topk_idx = split_cp_inputs(local_topk_idx, getattr(packed_seq_params, 'cu_seqlens_q', None), 1)
    # 3. sp slice.
    local_topk_idx = scatter_to_sequence_parallel_region(local_topk_idx.transpose(0, 1)).transpose(0, 1)
    return local_topk_idx


# ---------------------------------------------------------------------------
# Record / replay data exchange (public API mirrors the transformers module)
# ---------------------------------------------------------------------------


def _all_gather_seq(tokens: torch.Tensor, group: Any) -> torch.Tensor:
    """All-gather ``tokens`` along dim 0 in group-rank order (inverse of the contiguous SP scatter).

    Plain ``all_gather`` (not ``gather_from_sequence_parallel_region``) because recorded routing indices
    are integers with no autograd, and the SP scatter ``get_local_topk_idx_for_current_rank`` applies
    splits dim 0 into contiguous per-rank chunks -- concatenating in rank order undoes it exactly.
    """
    world = torch.distributed.get_world_size(group=group) if torch.distributed.is_initialized() else 1
    if world <= 1:
        return tokens
    gathered = [torch.empty_like(tokens) for _ in range(world)]
    torch.distributed.all_gather(gathered, tokens.contiguous(), group=group)
    return torch.cat(gathered, dim=0)


def get_router_replay_data(model: Any, batch_size: int = 1, packed_seq_params: Any = None) -> Optional[torch.Tensor]:
    """Collect this rank's recorded routing and rebuild the whole-model all-layer tensor.

    This is the exact inverse of the slice ``set_router_replay_data`` applies, so an R2 ``RECORD ->
    REPLAY`` round-trip -- where the recorded tensor travels back through the Ray driver and is replayed
    in a later forward -- reproduces the same local indices on every rank.

    Returns ``[batch_size, full_seq_len, num_layers, topk]`` where ``num_layers`` is the model's TOTAL
    transformer-layer count (dense-layer slots are zero-filled). That all-layer layout is the shared
    dev/sampler boundary format for both backends: vLLM reports routing across all transformer layers
    (dense included), and ``get_local_topk_idx_for_current_rank`` filters them back out on replay.
    Returns *None* when nothing was recorded.

    Scope guard (fail-loudly): a RECORD forward only runs the layers resident on this rank, so
    rebuilding the whole-model tensor under pipeline parallelism (PP>1) or virtual pipeline (VPP) would
    need a cross-stage gather that this port does not perform. Both raise rather than silently returning
    one stage's layers and corrupting replay; use ``router_replay_mode="R3"`` (routing delivered
    whole-model by the sampler, which needs no record forward) for PP>1 / VPP MoE runs.
    """
    _require_available()
    tf_config = _tf_config(model)
    from megatron.core import mpu
    from megatron.core.transformer.transformer_block import get_num_layers_to_build

    if mpu.get_pipeline_model_parallel_world_size() > 1 or tf_config.virtual_pipeline_model_parallel_size is not None:
        raise NotImplementedError(
            'Megatron R2 routing RECORD is only supported with pipeline parallelism disabled (PP=1, no '
            'VPP): a record forward sees just this stage\'s MoE layers, and rebuilding the whole-model '
            'routing tensor needs a cross-stage gather that is not wired. Use router_replay_mode="R3" '
            '(routing returned whole-model by the sampler) for PP>1 / VPP MoE runs.')

    router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config)
    local_layers = []
    for router in router_instances_list:
        if router.recorded_topk_idx is None:
            return None
        local_layers.append(router.recorded_topk_idx.to(torch.uint8))
    if not local_layers:
        return None
    num_local_moe = len(local_layers)
    local_tokens, topk = local_layers[0].shape[0], local_layers[0].shape[-1]
    assert local_tokens % batch_size == 0, (
        f'recorded routing token count {local_tokens} is not divisible by batch_size {batch_size}')
    local_seq = local_tokens // batch_size
    # [local_moe, bs*local_seq, topk] -> [bs, local_seq, local_moe, topk]
    local = torch.stack(local_layers, dim=0).reshape(num_local_moe, batch_size, local_seq,
                                                     topk).permute(1, 2, 0, 3).contiguous()
    # 1. inverse SP scatter (get_local_topk_idx scattered dim 0 of [seq, bs, ...] across the TP group).
    if getattr(tf_config, 'sequence_parallel', False) and mpu.get_tensor_model_parallel_world_size() > 1:
        local = _all_gather_seq(local.transpose(0, 1).contiguous(),
                                mpu.get_tensor_model_parallel_group()).transpose(0, 1).contiguous()
    # 2. inverse CP split (get_local_topk_idx load-balanced-split the seq dim, i.e. dim 1 here).
    if mpu.get_context_parallel_world_size() > 1:
        from twinkle.utils.torch_utils import gather_cp_load_balanced
        cu = getattr(packed_seq_params, 'cu_seqlens_q', None) if packed_seq_params is not None else None
        local = gather_cp_load_balanced(local, mpu.get_context_parallel_group(), seq_dim=1, cu_seqlens=cu)
    # local now: [bs, full_seq, num_local_moe, topk]
    full_seq = local.shape[1]
    # 3. inverse pp/hybrid index_select: scatter the local MoE layers back to their GLOBAL layer slots.
    num_layers = get_num_layers_to_build(tf_config)  # PP=1 & no VPP -> total transformer layers
    moe_layer_idx = [i for i in range(num_layers) if is_moe_layer(tf_config, i)]
    assert len(moe_layer_idx) == num_local_moe, (
        f'local MoE router count {num_local_moe} != model MoE layer count {len(moe_layer_idx)}')
    all_layer = local.new_zeros((batch_size, full_seq, num_layers, topk))
    for slot, global_idx in enumerate(moe_layer_idx):
        all_layer[:, :, global_idx, :] = local[:, :, slot, :]
    return all_layer


def set_router_replay_data(routed_experts: Optional[torch.Tensor],
                           model: Any,
                           packed_seq_params: Any = None) -> None:
    """Inject a whole-model ``routed_experts`` tensor as the replay target of this rank's local routers.

    ``routed_experts`` is ``[bs, seq_len, layers, topk]`` as reported across ALL transformer layers (vLLM
    emits routing for dense layers too -- see ``get_local_topk_idx_for_current_rank``). It is first sliced
    to this (pp, vp, cp, sp) rank's local MoE-layer / local-token shard, which leaves ``count`` local layer
    rows; the ``i``-th local router (``global_router_replay_instances[offset + i]``) then receives row
    ``i`` of that shard as its ``[total_seq, topk]`` target.

    Convention note (resolved from legacy swift's two call sites): the *logps* path
    (``swift/megatron/trainers/utils.py``) pre-slices via ``get_local_topk_idx_for_current_rank`` before
    distributing, which is the convention matching the real all-layer vLLM tensor and is what this port
    follows; the legacy *training* path (``grpo_trainer.forward_step``) instead indexed a full-global-layer
    tensor with ``[i + offset]`` and did no cp/sp slice. This port keeps the pre-slice convention so the
    index is ``[i]`` (offset already applied by the router-list slice) and cp/sp are handled, and is
    therefore correct for pp/vp/cp/sp alike. Mirrors the transformers-side
    ``set_router_replay_data(routed_experts, model)`` signature plus the ``packed_seq_params`` the Megatron
    cp slice needs.
    """
    _require_available()
    if routed_experts is None:
        return
    tf_config = _tf_config(model)
    local_topk_idx = get_local_topk_idx_for_current_rank(routed_experts, tf_config, packed_seq_params)
    if local_topk_idx is None:
        return
    # bs, seq_len, layer_num, topk -> layer_num, total_seq_len, topk
    layers_topk_idx_reshape = local_topk_idx.flatten(0, 1).transpose(0, 1).to(
        torch.cuda.current_device() if torch.cuda.is_available() else local_topk_idx.device)
    offset, count = get_local_layer_range(tf_config)
    router_instances_list = RouterReplay.global_router_replay_instances[offset:offset + count]
    for i, router in enumerate(router_instances_list):
        router.set_target_indices(layers_topk_idx_reshape[i].to(torch.int64))


def set_global_router_replay_action(action: Any) -> None:
    """Set *action* on every registered mcore router instance."""
    _require_available()
    RouterReplay.set_global_router_replay_action(action)


def clear_global_router_replay_action() -> None:
    """Reset the action to None on every registered mcore router instance."""
    _require_available()
    RouterReplay.clear_global_router_replay_action()


def clear_global_indices() -> None:
    """Clear recorded / target indices on every registered mcore router instance."""
    _require_available()
    RouterReplay.clear_global_indices()


class RouterReplayHelper:
    """Query router-replay state and locate the local ``RouterReplay`` instances for a model chunk."""

    @staticmethod
    def get_micro_batch_router_list(tf_config: Any, vp_rank: Optional[int] = None) -> List[Any]:
        """Return the ``RouterReplay`` instances for the current micro-batch and local (pp, vp) range.

        When virtual pipeline (VPP) is enabled, the local range for the PP rank is expanded to include all
        VP stages. The returned slice is taken from ``RouterReplay.global_router_replay_instances``.
        """
        _require_available()
        offset, count = get_local_layer_range(tf_config, vp_rank)
        return RouterReplay.global_router_replay_instances[offset:offset + count]

    @staticmethod
    def is_r2_record_action(tf_config: Any, vp_rank: Optional[int] = None) -> bool:
        """True when the current action is RECORD (the R2 record phase) for the local routers."""
        _require_available()
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return bool(router_instances_list
                    and router_instances_list[0].router_replay_action == RouterReplayAction.RECORD)

    @staticmethod
    def is_replay_forward_action(tf_config: Any, vp_rank: Optional[int] = None) -> bool:
        """True when the current action is REPLAY_FORWARD for the local routers."""
        _require_available()
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return bool(router_instances_list
                    and router_instances_list[0].router_replay_action == RouterReplayAction.REPLAY_FORWARD)

    @staticmethod
    def is_replay_backward_action(tf_config: Any, vp_rank: Optional[int] = None) -> bool:
        """True when the current action is REPLAY_BACKWARD for the local routers."""
        _require_available()
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return bool(router_instances_list
                    and router_instances_list[0].router_replay_action == RouterReplayAction.REPLAY_BACKWARD)


def prepare_replay_forward(action: Any, model: Any, routed_experts: Optional[torch.Tensor],
                           packed_seq_params: Any = None) -> None:
    """Per-micro-batch pre-forward routing-replay step (mirrors legacy ``grpo_trainer.forward_step``).

    Under 1F1B/interleaved scheduling a previous micro-batch leaves its routers in ``REPLAY_BACKWARD``
    (set after its forward, consumed by its backward recompute), so reset them to ``REPLAY_FORWARD``
    first, then load *this* micro-batch's replay target. No-op when *action* is None (replay disabled for
    this forward).
    """
    if action is None:
        return
    _require_available()
    tf_config = _tf_config(model)
    if RouterReplayHelper.is_replay_backward_action(tf_config):
        for router in RouterReplayHelper.get_micro_batch_router_list(tf_config):
            router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    if RouterReplayHelper.is_replay_forward_action(tf_config):
        set_router_replay_data(routed_experts, model, packed_seq_params)


def finish_replay_forward(action: Any,
                          model: Any,
                          packed_seq_params: Any = None,
                          recorded_sink: Optional[List[torch.Tensor]] = None,
                          batch_size: int = 1) -> None:
    """Per-micro-batch post-forward routing-replay step.

    ``REPLAY_FORWARD`` -> flip the routers to ``REPLAY_BACKWARD`` so the backward activation recompute
    replays the SAME routing (mcore pops the target that ``set_target_indices`` pushed onto
    ``replay_backward_list``). ``RECORD`` -> collect this micro-batch's recorded routing into
    *recorded_sink* (a list the caller concatenates onto its output). No-op when *action* is None.
    """
    if action is None:
        return
    _require_available()
    tf_config = _tf_config(model)
    if RouterReplayHelper.is_replay_forward_action(tf_config):
        for router in RouterReplayHelper.get_micro_batch_router_list(tf_config):
            router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
    elif RouterReplayHelper.is_r2_record_action(tf_config):
        recorded = get_router_replay_data(model, batch_size, packed_seq_params)
        if recorded is not None and recorded_sink is not None:
            recorded_sink.append(recorded)


def apply_router_replay_patch(model: Any = None) -> None:
    """Patch ``MoEAlltoAllTokenDispatcher.preprocess`` so all-to-all split sizes survive replayed routing.

    Idempotent and global (the patch is on the dispatcher class, so *model* is accepted only for signature
    parity with the transformers-side ``apply_router_replay_patch(model)``). With routing replay, duplicate
    indices in ``top_indices`` can make ``routing_map.sum() < num_tokens * topk``, which would otherwise
    desync the all-to-all split sizes.
    """
    _require_available()
    if MoEAlltoAllTokenDispatcher is None or hasattr(MoEAlltoAllTokenDispatcher, '_preprocess_patched'):
        return
    logger.info('Applying Megatron MoE router replay patch...')
    original_preprocess = MoEAlltoAllTokenDispatcher.preprocess

    def patched_preprocess(self, routing_map):
        result = original_preprocess(self, routing_map)
        # With router replay, duplicate indices can reduce the actual routed token count, so derive it from
        # the routing map instead of assuming num_tokens * topk.
        if (getattr(self.config, 'moe_enable_routing_replay', False) and not self.drop_and_pad
                and self.config.moe_expert_capacity_factor is None
                and not (getattr(self.config, 'moe_router_padding_for_quantization', None)
                         or getattr(self.config, 'moe_router_padding_for_fp8', None))):
            self.num_out_tokens = int(routing_map.sum().item())
        return result

    MoEAlltoAllTokenDispatcher.preprocess = patched_preprocess
    MoEAlltoAllTokenDispatcher._preprocess_patched = True
