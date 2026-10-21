# Copyright (c) ModelScope Contributors. All rights reserved.
import contextlib
import gc

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from typing import Any, Dict, List, Literal, Optional

from twinkle import DeviceMesh, Platform, torch_util
from twinkle.utils import get_logger

logger = get_logger()


def finalize_model_grads_for_lora(model, *args, **kwargs):
    """Only enter Megatron native finalize when the wrapped model has sync capability.

    In single-rank/no-op wrap cases Twinkle attaches ``ddp_config`` to the bare
    module for optimizer compatibility, but that does not mean the model really
    implements ``finish_grad_sync()``. Native Megatron finalize ultimately calls
    that method, so we gate by runtime capability instead of config metadata.
    """
    from megatron.core.distributed import DistributedDataParallel as MegatronDDP
    from megatron.core.distributed import finalize_model_grads as _native_finalize_model_grads
    from peft import PeftModel as _PeftModel

    def _get_base_model(m):
        if isinstance(m, _PeftModel):
            return _get_base_model(m.base_model.model)
        return m

    base_model = _get_base_model(model[0])
    if isinstance(base_model, MegatronDDP) or hasattr(base_model, 'finish_grad_sync'):
        return _native_finalize_model_grads(model, *args, **kwargs)
    return None


# transformer_engine has no per-call FlashAttention version selector; the only lever is its
# module-level availability flags. Pinning asserts the requested build is installed, then
# switches the others OFF so TE's dispatcher can only pick the requested one (legacy does the
# same, megatron_args.py:905-915).
_FLASH_PIN_FLAGS = {2: 'is_installed', 3: 'v3_is_installed', 4: 'v4_is_installed'}
_FLASH_PIN_PREFIXES = ('flash_attention_', 'flash_')


def _flash_version_pin(attn_impl: Optional[str]) -> Optional[int]:
    """The FlashAttention major version ``attn_impl`` pins, or None when it pins nothing.

    Accepts both spellings (Megatron ``flash_3`` / transformers ``flash_attention_3``) so the
    same intent resolves identically. Unversioned names ('flash', 'flash_attn', 'sdpa', ...)
    -> None.
    """
    if attn_impl is None:
        return None
    key = str(attn_impl).lower()
    for prefix in _FLASH_PIN_PREFIXES:
        if key.startswith(prefix) and key[len(prefix):].isdigit():
            return int(key[len(prefix):])
    return None


def _apply_flash_version_pin(attn_impl: Optional[str]) -> Optional[int]:
    """Force transformer_engine to use exactly the FlashAttention version ``attn_impl`` pins.

    Returns the pinned version, or None when nothing was pinned (TE keeps its own choice).
    MUST run in the process that builds the model: these are transformer_engine module
    globals, so a driver-side call would not reach a Ray worker.
    """
    version = _flash_version_pin(attn_impl)
    if version is None:
        return None
    from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils as fa_utils

    if version not in _FLASH_PIN_FLAGS:
        raise NotImplementedError(
            f'attn_impl={attn_impl!r} pins FlashAttention v{version}, which transformer_engine has no '
            f'availability flag for. Supported: {sorted(_FLASH_PIN_FLAGS)}.')
    # Only flags this TE build actually defines: v4_is_installed is absent on older versions, and a
    # bare getattr would surface as AttributeError instead of an actionable message.
    present = {v: flag for v, flag in _FLASH_PIN_FLAGS.items() if hasattr(fa_utils, flag)}
    if version not in present:
        raise ValueError(f'attn_impl={attn_impl!r} requests flash-attn v{version}, but the installed '
                         f'transformer_engine has no {_FLASH_PIN_FLAGS[version]!r} flag, so this version cannot be '
                         f'selected. Versions this TE build can pin: {sorted(present)}.')
    installed = {v: getattr(fa_utils, flag) for v, flag in present.items()}
    if not installed[version]:
        raise ValueError(f'attn_impl={attn_impl!r} requests flash-attn v{version}, which is not installed. '
                         f'Detected: ' + ', '.join(f'FA{v}={state}' for v, state in sorted(installed.items())))
    for other, flag in present.items():
        if other != version:
            setattr(fa_utils, flag, False)
    return version


class MegatronStrategy:

    #: ddp_config keys that select a sharded (Megatron-FSDP) data-parallel wrapper instead of plain
    #: DDP. ``use_custom_fsdp`` is megatron's own deprecated alias -- its arguments.py still derives
    #: it from ``use_megatron_fsdp`` -- so a config carrying either key means the same thing here.
    _FSDP_KEYS = ('use_megatron_fsdp', 'use_custom_fsdp')

    @classmethod
    def uses_fsdp(cls, ddp_config: Optional[Dict[str, Any]]) -> bool:
        """Whether this ddp_config asks for Megatron-FSDP rather than DDP.

        Public because the verdict is needed at several points where no strategy instance exists
        yet: the wrapper class choice, the checkpoint sharding type and the optimizer wiring all
        have to agree, and they are decided at different points in the run.
        """
        ddp_config = ddp_config or {}
        return any(ddp_config.get(key) for key in cls._FSDP_KEYS)

    def __init__(
        self,
        model_dir,
        device_mesh: Optional[DeviceMesh] = None,
        use_distributed_optimizer: bool = True,
        mixed_precision: Literal['no', 'fp16', 'bf16'] = 'bf16',
        seed: int = 42,
        variable_seq_lengths: bool = True,
        config: PretrainedConfig = None,
        ddp_config: Dict[str, Any] = None,
        bridge_backend='mcore-bridge',
        align_grad_reduce: bool = True,
        nccl_comm_warmup: bool = False,
        attn_impl: Optional[str] = None,
        **kwargs,
    ):
        from megatron.core import mpu
        self.device_mesh = device_mesh
        self.use_distributed_optimizer = use_distributed_optimizer
        self.mixed_precision = mixed_precision
        self.model_dir = model_dir
        self.seed = seed
        self.variable_seq_lengths = variable_seq_lengths
        self.ddp_config = ddp_config or {}
        # The bridge backend owns model construction: get_model_config / create_megatron_model below
        # delegate to it. Accepted as a NAME string or a live instance -- the string is what survives
        # the Ray worker boundary (@remote_class forwards constructor kwargs, not built objects),
        # while an instance allows local/test injection. Resolved before get_model_config runs.
        if isinstance(bridge_backend, str):
            from ..bridge import resolve_bridge_backend
            self._backend = resolve_bridge_backend(bridge_backend)
        else:
            self._backend = bridge_backend
        self._align_grad_reduce = align_grad_reduce
        # FlashAttention version pin: flips transformer_engine MODULE GLOBALS, so it must run in this
        # (the model-building) process -- in Ray mode the worker, not the driver. attn_impl is a
        # caller field name ModelConfig does not have, so it is consumed here and never forwarded
        # into the config; the kernel choice arrives separately as attention_backend.
        pinned = _apply_flash_version_pin(attn_impl)
        if pinned is not None:
            logger.info(f'Forcing Flash Attention v{pinned} as the attention backend.')
        if config is None:
            from transformers import AutoConfig
            self.hf_config = AutoConfig.from_pretrained(self.model_dir, trust_remote_code=True)
        else:
            self.hf_config = config
        num_experts = getattr(self.hf_config, 'num_experts', getattr(self.hf_config, 'num_local_experts', None))
        if (num_experts not in (None, 0, 1) and (self.device_mesh.tp_world_size or 1) > 1
                and not getattr(self.device_mesh, 'sequence_parallel', False)):
            # Megatron 0.15.3 requires sequence parallelism for MoE training when
            # tensor parallelism is enabled. Keep this policy in the framework so
            # cookbook scripts do not need to know a model-family-specific
            # runtime constraint just to launch a valid MoE run.
            self.device_mesh.sequence_parallel = True
            logger.info('Auto-enabled sequence_parallel for MoE model with tensor parallelism.')
        if 'overlap_grad_reduce' not in self.ddp_config:
            self.ddp_config['overlap_grad_reduce'] = False
        if 'overlap_param_gather' not in self.ddp_config:
            self.ddp_config['overlap_param_gather'] = False
        if 'align_param_gather' not in self.ddp_config:
            self.ddp_config['align_param_gather'] = False
        if 'grad_reduce_in_fp32' not in self.ddp_config:
            self.ddp_config['grad_reduce_in_fp32'] = True

        # Determine params_dtype and activation checkpointing kwargs
        params_dtype = torch.bfloat16
        if self.mixed_precision == 'fp16':
            params_dtype = torch.float16
        elif self.mixed_precision == 'no':
            params_dtype = torch.float32
        self._params_dtype = params_dtype

        vpp_size = self.device_mesh.vpp_size
        if vpp_size in (0, 1):
            vpp_size = None

        parallel_kwargs = {
            'tensor_model_parallel_size': self.device_mesh.tp_world_size or 1,
            'pipeline_model_parallel_size': self.device_mesh.pp_world_size or 1,
            'context_parallel_size': self.device_mesh.cp_world_size or 1,
            'expert_model_parallel_size': self.device_mesh.ep_size or 1,
            'expert_tensor_parallel_size': self.device_mesh.etp_world_size or 1,
            'virtual_pipeline_model_parallel_size': vpp_size,
        }
        if not vpp_size:
            # non-interleave does not support overlap_p2p_comm
            kwargs['overlap_p2p_comm'] = False
        if 'overlap_p2p_comm' not in kwargs:
            kwargs['overlap_p2p_comm'] = True
            kwargs['batch_p2p_comm'] = not kwargs['overlap_p2p_comm']

        init_kwargs = {
            'order': self.device_mesh.order,
            **parallel_kwargs,
        }
        if Platform.device_prefix() == 'npu':
            init_kwargs['create_gloo_process_groups'] = True
        mpu.initialize_model_parallel(**init_kwargs)
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        model_parallel_cuda_manual_seed(self.seed)
        self.config = self.get_model_config(self.hf_config, parallel_kwargs, **kwargs)
        self._finalize_quantized_param_config()
        self._check_fsdp()
        if nccl_comm_warmup:
            self._warmup_communicators()

    def _check_fsdp(self):
        """Reject Megatron-FSDP combinations that cannot work.

        Raised rather than fixed up, matching ``_finalize_quantized_param_config``: silently
        rewriting a user's distributed configuration is how a run ends up doing something other than
        what its config says, which is the failure mode this whole area keeps producing.
        """
        if not self.uses_fsdp(self.ddp_config):
            return
        if not self.use_distributed_optimizer:
            # Megatron-FSDP shards the parameters, so the optimizer has to own matching shards of the
            # master weights; the non-distributed optimizer keeps whole tensors and has no path that
            # writes back into a sharded parameter.
            raise ValueError('Megatron-FSDP requires use_distributed_optimizer=True: it shards the '
                             'parameters, and only the distributed optimizer maintains the matching '
                             'master-weight shards. Drop use_megatron_fsdp, or enable the distributed optimizer.')
        if self.device_mesh is not None and getattr(self.device_mesh, 'cp_world_size', 1) > 1:
            # megatron asserts the same pairing on its own CLI ('Hybrid context parallelism not
            # supported with Megatron FSDP'). Checked here so it fails while the config is being
            # built, rather than part-way through wrapping the model on every rank.
            raise ValueError('Megatron-FSDP does not support context parallelism '
                             f'(cp_world_size={self.device_mesh.cp_world_size}). Use DDP for a CP run.')

    def _finalize_quantized_param_config(self):
        """Keep the DDP param-gather flags in step with the model's quantized-parameter config.

        Quantized (FP8/FP4) parameters are described twice, on two config objects megatron never
        cross-checks: ``TransformerConfig.fp8_param`` / ``fp4_param`` decide that the parameters are
        BUILT quantized, while ``DistributedDataParallelConfig.fp8_param_gather`` /
        ``fp4_param_gather`` decide that the distributed optimizer writes its FP32 master shards back
        INTO them, in ``DistributedOptimizer._copy_main_params_to_model_params``.

        Setting only the first is a silent no-op rather than a slow path. That method dispatches on
        the DDP flags alone, and its generic fallback then skips exactly the parameters the skipped
        branch was supposed to handle ("FP8 params are quantized in the above quantize_param_shard
        function" / the matching NVFP4 comment). So the master weights advance every step while the
        model parameters stay at their loaded values: no error, no warning, and a loss curve that
        simply never moves.

        Derived rather than merely validated because there is no configuration in which one wants the
        pair split; an explicitly requested value is still honoured, and only an absent one is filled.
        """
        for config_attr, ddp_key in (('fp8_param', 'fp8_param_gather'), ('fp4_param', 'fp4_param_gather')):
            if not getattr(self.config, config_attr, False):
                continue
            if self.ddp_config.get(ddp_key) is False:
                raise ValueError(
                    f'{config_attr}=True builds the parameters quantized, but {ddp_key}=False stops the '
                    f'distributed optimizer from ever writing the updated master weights back into them, so '
                    f'the model would not train at all. Drop {ddp_key}=False, or turn off {config_attr}.')
            self.ddp_config[ddp_key] = True
            if not self.use_distributed_optimizer:
                # The quantize-back step exists only on DistributedOptimizer; the plain optimizer has
                # no equivalent, so the parameters would again never be updated.
                raise ValueError(f'{config_attr}=True requires use_distributed_optimizer=True: quantized '
                                 'parameters are updated by re-quantizing the distributed optimizer\'s master '
                                 'shards, which is the only code path that writes them.')

    @property
    def sequence_parallel(self) -> bool:
        """Read from device_mesh so auto-enable in args.py is visible."""
        return getattr(self.device_mesh, 'sequence_parallel', False)

    def init_kwargs(self) -> Dict[str, Any]:
        # Parallelism is configured through the megatron model config, not at construction time.
        return {}

    @property
    def bridge(self):
        return self.config.bridge

    @property
    def backend(self):
        """The BridgeBackend this strategy delegates model construction to."""
        return self._backend

    @property
    def params_type(self) -> torch.dtype:
        if self._params_dtype is not None:
            dtype_map = {
                'fp32': torch.float32,
                'fp16': torch.float16,
                'bf16': torch.bfloat16,
            }
            return dtype_map.get(self._params_dtype, torch.bfloat16)

        if self.mixed_precision == 'bf16':
            return torch.bfloat16
        elif self.mixed_precision == 'fp16':
            return torch.float16
        return torch.float32

    def _check_device_mesh(self):
        from megatron.core import parallel_state as mpu

        assert self.device_mesh.dp_world_size == mpu.get_data_parallel_world_size()
        assert self.device_mesh.dp_rank == mpu.get_data_parallel_rank()

        # Only validate world sizes match
        if self.device_mesh.tp_world_size > 1:
            assert self.device_mesh.tp_world_size == mpu.get_tensor_model_parallel_world_size()
            assert self.device_mesh.tp_rank == mpu.get_tensor_model_parallel_rank()

        if self.device_mesh.pp_world_size > 1:
            assert self.device_mesh.pp_world_size == mpu.get_pipeline_model_parallel_world_size()
            assert self.device_mesh.pp_rank == mpu.get_pipeline_model_parallel_rank()
            assert self.device_mesh.is_pp_last_rank() == mpu.is_pipeline_last_stage()
            assert self.device_mesh.is_pp_first_rank() == mpu.is_pipeline_first_stage()

        if self.device_mesh.cp_world_size > 1:
            assert self.device_mesh.cp_world_size == mpu.get_context_parallel_world_size()
            assert self.device_mesh.cp_rank == mpu.get_context_parallel_rank()

        if self.device_mesh.vpp_size is not None and self.device_mesh.vpp_size > 1:
            assert self.device_mesh.vpp_size == mpu.get_virtual_pipeline_model_parallel_world_size()

    def wrap_model(
        self,
        model: List[nn.Module],
    ) -> List[nn.Module]:
        if self.device_mesh.world_size <= 1:
            from megatron.core.distributed import DistributedDataParallelConfig
            ddp_config = DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                use_distributed_optimizer=False,
            )
            for m in model:
                if not hasattr(m, 'ddp_config'):
                    m.ddp_config = ddp_config
            return model

        self._check_device_mesh()
        return self._wrap_with_megatron_ddp(model, self.use_distributed_optimizer, self.ddp_config)

    def unwrap_model(self, model: List[nn.Module]) -> List[nn.Module]:
        from megatron.core.distributed import DistributedDataParallel as MegatronDDP
        from megatron.core.distributed import FullyShardedDataParallel as MegatronFSDP
        from megatron.core.transformer.module import Float16Module
        from torch.nn.parallel import DistributedDataParallel as TorchDDP
        _models = []
        for _model in model:
            # Unwrap DDP first
            while isinstance(_model, (MegatronDDP, MegatronFSDP, TorchDDP, Float16Module)):
                _model = _model.module
            _models.append(_model)
        return _models

    def get_sharded_sd_metadata(self) -> Dict[str, Any]:
        """Metadata describing how the distributed optimizer's state is sharded on disk.

        Lives on the strategy because it has to agree with the wrapper class chosen in
        ``_wrap_with_megatron_ddp``, and those are two decisions made at very different times -- the
        model is wrapped at startup, the metadata is read on every save and every resume. Keeping
        them in one class is what stops them drifting apart into a checkpoint whose declared layout
        does not match the one actually written.

        Megatron-FSDP shards the optimizer state as DTensors, which is a different on-disk layout
        from the distributed optimizer's own; megatron only accepts ``fsdp_dtensor`` for FSDP runs and
        the reshardable type for the rest.
        """
        metadata = {'singleton_local_shards': False, 'chained_optim_avoid_prefix': True}
        if self.uses_fsdp(self.ddp_config):
            metadata['distrib_optim_sharding_type'] = 'fsdp_dtensor'
        else:
            metadata['distrib_optim_sharding_type'] = 'dp_reshardable'
        return metadata

    def finish_param_config(self, model: List[nn.Module], optimizer: Any):
        self.config.grad_scale_func = getattr(optimizer, 'scale_loss') if optimizer is not None else None
        ddp_config = self.ddp_config
        if ddp_config['overlap_grad_reduce']:
            assert self.config.no_sync_func is None, (
                'When overlap_grad_reduce is True, config.no_sync_func must be None; '
                'a custom no_sync_func is not supported when overlapping grad-reduce')
            self.config.no_sync_func = [model_chunk.no_sync for model_chunk in model]  # noqa
            if len(model) == 1:
                self.config.no_sync_func = self.config.no_sync_func[0]  # noqa
            self.config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]  # noqa
            if len(model) == 1:
                self.config.grad_sync_func = self.config.grad_sync_func[0]  # noqa
            if not self._align_grad_reduce:
                # align_grad_reduce=False opts out of bucket-aligned reduction. grad_sync_func is the
                # hook driving the aligned overlap, so clear it (legacy does the same). Placed inside
                # the overlap_grad_reduce branch -- and before the overlap_param_gather block, which
                # can early-return -- so it always runs when there is something to undo.
                self.config.grad_sync_func = None
        if ddp_config['overlap_param_gather'] and ddp_config['align_param_gather']:
            # Only DDP exposes start_param_sync. Megatron-FSDP has no equivalent because it drives
            # its own parameter all-gathers from the module hooks rather than being prompted by the
            # forward-backward schedule, so there is nothing to hand over -- and reaching for the
            # attribute anyway would be an AttributeError at wrap time.
            missing = [m for m in model if not hasattr(m, 'start_param_sync')]
            if missing:
                logger.info('Skipping config.param_sync_func: the data-parallel wrapper '
                            f'({type(missing[0]).__name__}) does not implement start_param_sync, which is '
                            'a DDP-only hook. Its parameter gathering is driven internally instead.')
                return
            self.config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]  # noqa
            if len(model) == 1:
                self.config.param_sync_func = self.config.param_sync_func[0]  # noqa

    @staticmethod
    def _wrap_with_megatron_ddp(
        model: List[nn.Module],
        use_distributed_optimizer: bool,
        ddp_config: Dict[str, Any],
    ) -> List[nn.Module]:
        from megatron.core.distributed import DistributedDataParallel as MegatronDDP
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.transformer import TransformerConfig
        from megatron.core.transformer.module import Float16Module

        # Megatron-FSDP is a sibling of DDP, not a subclass of it, but it takes the same three
        # constructor arguments and the same DistributedDataParallelConfig, so the choice is only
        # about which class to instantiate. It has to be made from ddp_config rather than left to
        # ``use_megatron_fsdp`` flowing into the config object: the config would accept the flag and
        # the run would still be wrapped in DDP -- FSDP silently never engaged, no error anywhere.
        if MegatronStrategy.uses_fsdp(ddp_config):
            from megatron.core.distributed import FullyShardedDataParallel as data_parallel_cls
        else:
            data_parallel_cls = MegatronDDP

        wrapped_models = []
        for _model in model:
            _model = MegatronStrategy._move_model_to_gpu(_model)
            config: TransformerConfig = _model.config  # noqa

            if not isinstance(model, Float16Module) and (config.fp16 or config.bf16):
                _model = Float16Module(config, _model)

            ddp_config_cls = DistributedDataParallelConfig(
                **ddp_config,
                use_distributed_optimizer=use_distributed_optimizer,
            )
            wrapped_model = data_parallel_cls(
                config=config,
                ddp_config=ddp_config_cls,
                module=_model,
            )

            # Broadcast params from data parallel src rank
            # In torchrun mode, all ranks enter here simultaneously, so this works
            wrapped_model.broadcast_params()
            wrapped_models.append(wrapped_model)

        return wrapped_models

    def reduce_loss(self, local_loss, local_count, logits, logps):
        count = local_count.clamp(min=1).to(torch.int64)
        cp_size = self.device_mesh.cp_world_size or 1
        grad_count = (count // cp_size).clamp(min=1) if cp_size > 1 else count
        # The reported loss MUST be a real copy (.clone()), not a storage-sharing .detach() view of
        # local_loss. This 3-tuple is the (loss, num_tokens, loss_reduced) contract Megatron-core's
        # forward_step_calc_loss consumes: it takes the FIRST element as output_tensor and scales it
        # IN PLACE for gradient accumulation -- `output_tensor /= clamp(num_tokens); /= num_microbatches`
        # (schedules.py). local_loss IS that first element, so a bare local_loss.detach() in the dict
        # shares its storage and gets silently divided by num_microbatches (and num_tokens) too, corrupting
        # the logged/metric loss. It only showed up at num_microbatches>1 (a single microbatch divides by 1)
        # and only for losses that report num_tokens=0 (mean-reduced preference/RL losses like DPO, whose
        # per-microbatch value is already the batch mean); per-token-sum losses hide it because their
        # num_tokens scaling is folded back out downstream. clone() snapshots the true value before mcore
        # touches the backward tensor. logits/logps are separate tensors mcore never scales, so detach
        # suffices for them.
        return local_loss, grad_count, {
            'loss': local_loss.detach().clone(),
            'logits': logits.detach() if logits is not None else None,
            'logps': logps.detach() if logps is not None else None,
            'num_tokens': count
        }

    def get_model_config(
        self,
        hf_config: PretrainedConfig,
        parallel_kwargs: Dict[str, Any],
        **kwargs,
    ):
        return self._backend.build_model_config(hf_config, parallel_kwargs, self, **kwargs)

    def create_megatron_model(
        self,
        load_weights: bool = True,
    ) -> List[nn.Module]:
        return self._backend.create_model(
            self.config, self.model_dir, load_weights=load_weights, move_to_gpu=self._move_model_to_gpu)

    @staticmethod
    def _move_model_to_gpu(model: nn.Module) -> nn.Module:
        model = model.to(Platform.get_local_device())
        torch_util.synchronize()
        return model

    @staticmethod
    def _warmup_communicators() -> None:
        """Run a dummy all-reduce on every NCCL communicator so the first real step does not pay the
        connection-setup cost and skew initial timings. Opt-in via ``nccl_comm_warmup``; requires mpu
        to be initialized (it is, by __init__, before this is called)."""
        from megatron.core import mpu

        dummy = torch.zeros(1, device=torch_util.get_current_device())
        warmed = 0
        for getter, kwargs in (
            (mpu.get_data_parallel_group, {
                'with_context_parallel': True
            }),
            (mpu.get_data_parallel_group, {}),
            (mpu.get_context_parallel_group, {}),
            (mpu.get_tensor_model_parallel_group, {}),
            (mpu.get_pipeline_model_parallel_group, {}),
            (mpu.get_model_parallel_group, {}),
            (mpu.get_embedding_group, {}),
            (mpu.get_position_embedding_group, {}),
        ):
            try:
                groups = getter(**kwargs)
            except (AssertionError, ValueError, TypeError):
                continue
            for group in groups if isinstance(groups, list) else [groups]:
                if group is not None:
                    torch.distributed.all_reduce(dummy, group=group)
                    warmed += 1
        torch_util.synchronize()
        logger.info(f'NCCL communicator warm-up done ({warmed} groups).')

    @contextlib.contextmanager
    def transactional_grad_schedule(self, model_chunks: List[nn.Module], *, forward_only: bool,
                                    transactional: bool, loss_weight: float, finalize_gradients: bool,
                                    completing_pending: bool, global_normalizer, **kwargs):
        """Temporarily rewrite this strategy's gradient hooks for one schedule, then restore them.

        The mcore ``config`` object is shared by every later step on this worker, so any
        per-call surgery (a scaled loss, a no-sync pass, a deferred finalize) must be
        undone even when the schedule raises. Owning that invariant here keeps the model
        layer from reaching into ``strategy.config`` internals: the model only decides
        WHAT this pass is (the keyword flags), this strategy decides HOW the hooks bend.

        Args:
            model_chunks: The model's chunks; those exposing ``no_sync`` enter it for a
                deferred-finalize pass (DDP bucket traffic pauses with the hooks).
            forward_only: A log-prob pass never touches the hooks.
            transactional: Scale the loss by ``loss_weight / normalizer / ga_steps``.
            loss_weight: Caller's per-pass loss weight (unscaled).
            finalize_gradients: False defers: gradients accumulate without syncing.
            completing_pending: True runs the stored finalize with ``num_tokens=None``.
            global_normalizer: Token normalizer the transactional scale divides by.
        """
        config = self.config
        original_grad_scale_func = config.grad_scale_func
        original_no_sync_func = config.no_sync_func
        original_grad_sync_func = config.grad_sync_func
        original_finalize_model_grads_func = config.finalize_model_grads_func

        scale = float(loss_weight)
        if transactional:
            gradient_accumulation_steps = int(kwargs.get('gradient_accumulation_steps') or 1)
            if gradient_accumulation_steps < 1:
                raise ValueError('gradient_accumulation_steps must be positive')
            scale = scale / global_normalizer.clamp(min=1) / gradient_accumulation_steps

        def transaction_grad_scale(value):
            if original_grad_scale_func is not None:
                value = original_grad_scale_func(value)
            return value * scale

        def disabled_no_sync():
            return contextlib.nullcontext()

        def disabled_grad_sync(*args, **kwargs):
            return None

        def disabled_finalize(*args, **kwargs):
            return None

        def finalize_accumulated_gradients(model, num_tokens=None, **finalize_kwargs):
            return original_finalize_model_grads_func(model, None, **finalize_kwargs)

        outer_no_sync = contextlib.ExitStack()
        try:
            with outer_no_sync:
                if not forward_only and (transactional or loss_weight != 1.0):
                    config.grad_scale_func = transaction_grad_scale
                if not forward_only and not finalize_gradients:
                    for model_chunk in model_chunks:
                        if hasattr(model_chunk, 'no_sync'):
                            outer_no_sync.enter_context(model_chunk.no_sync())
                    if isinstance(original_no_sync_func, list):
                        config.no_sync_func = [disabled_no_sync for _ in original_no_sync_func]
                    else:
                        config.no_sync_func = disabled_no_sync
                    if isinstance(original_grad_sync_func, list):
                        config.grad_sync_func = [disabled_grad_sync for _ in original_grad_sync_func]
                    else:
                        config.grad_sync_func = disabled_grad_sync
                    config.finalize_model_grads_func = disabled_finalize
                elif completing_pending and original_finalize_model_grads_func is not None:
                    config.finalize_model_grads_func = finalize_accumulated_gradients
                yield
        finally:
            config.grad_scale_func = original_grad_scale_func
            config.no_sync_func = original_no_sync_func
            config.grad_sync_func = original_grad_sync_func
            config.finalize_model_grads_func = original_finalize_model_grads_func

    @torch.no_grad()
    def offload_to_cpu(self, model: List[nn.Module], optimizer: Any = None) -> None:
        """Release this rank's training memory so something else can use the device.

        Needed for colocation, where a sampler shares the GPU with the trainer and the two do not
        fit at once. The buffer-level work is Megatron's own ``offload_to_cpu`` -- parameters and
        gradients live in flat per-bucket buffers, not on the modules, so freeing them means
        emptying that storage, and Megatron already knows how.

        Reversed by :meth:`reload_to_gpu`. Both read what is currently resident rather than a flag,
        so a repeated call is a no-op.
        """
        # Buffers are about to have their storage freed; make sure nothing is still writing to it.
        # Megatron's own offload_grad_buffers opens the same way.
        torch_util.synchronize()
        for chunk in model:
            buffers = self._flat_buffers(chunk)
            for buffer in buffers:
                buffer.offload_to_cpu()
            self._move_loose_params(chunk, buffers, 'cpu')
        if optimizer is not None:
            self._move_optimizer_state(optimizer, 'cpu')
        gc.collect()
        torch_util.empty_cache()

    @torch.no_grad()
    def reload_to_gpu(self, model: List[nn.Module], optimizer: Any = None) -> None:
        """Bring back what :meth:`offload_to_cpu` released."""
        device = Platform.get_local_device()
        for chunk in model:
            buffers = self._flat_buffers(chunk)
            for buffer in buffers:
                buffer.reload_from_cpu()
            self._move_loose_params(chunk, buffers, device)
        if optimizer is not None:
            self._move_optimizer_state(optimizer, device)
        gc.collect()
        torch_util.empty_cache()

    @staticmethod
    def _flat_buffers(chunk: nn.Module) -> List[Any]:
        """The flat parameter/gradient buckets owning this chunk's trainable tensors, if any.

        This cannot duck-type on the attribute name: ``nn.Module`` already defines a ``buffers()``
        method, so an unwrapped chunk would hand back a bound method instead of a list.
        """
        from megatron.core.distributed import DistributedDataParallel as MegatronDDP
        from megatron.core.distributed import FullyShardedDataParallel as MegatronFSDP
        if isinstance(chunk, MegatronDDP):
            return [*chunk.buffers, *chunk.expert_parallel_buffers]
        if isinstance(chunk, MegatronFSDP):
            raise NotImplementedError(
                'Offloading a Megatron-FSDP model is not supported: it holds its parameters in '
                '`param_and_grad_buffer`, which is not the per-bucket structure DDP uses. Freeing '
                'only the parts this method understands would leave most of the training memory '
                'resident while reporting success, which defeats the purpose.')
        return []

    @staticmethod
    def _move_loose_params(chunk: nn.Module, buffers: List[Any], device: Any) -> None:
        """Move the parameters no flat buffer holds.

        Three ways a parameter ends up outside them. It is frozen -- a LoRA base weight, which DDP
        keeps out of its buckets because it never needs a gradient. Or the chunk was never wrapped:
        a reference model, or any model on a single rank, where ``wrap_model`` skips DDP. Or the run
        has no distributed optimizer, in which case Megatron allocates ``grad_data`` but leaves
        ``param_data`` as None -- only a distributed optimizer needs parameters pooled into a flat
        buffer, so without one they stay on the module and the buckets hold gradients alone.

        That last case is why this asks the buffers what they actually hold instead of whether they
        exist: gradient-only buckets would otherwise look like full coverage and the parameters
        would silently stay on the device.
        """
        params_are_pooled = any(buffer.param_data is not None for buffer in buffers)
        for param in chunk.parameters():
            if params_are_pooled and param.requires_grad:
                continue
            param.data = param.data.to(device, non_blocking=True)
            if param.grad is not None:
                param.grad = param.grad.to(device, non_blocking=True)

    @staticmethod
    def _move_optimizer_state(optimizer: Any, device: Any) -> None:
        """Move an optimizer's tensor state between host and device.

        Megatron has ``MegatronOptimizer.offload_to_cpu``/``restore_from_cpu`` for this, but they
        hardcode ``.cuda()`` and ``torch.cuda.empty_cache()``, so they cannot serve the NPU backend
        Twinkle also supports. This is the same walk with the device left as a parameter.

        Walking every tensor in ``state`` rather than naming ``exp_avg``/``exp_avg_sq`` holds for any
        optimizer keeping its state on one device -- but not for a heterogeneous one that
        deliberately splits its state across host and device.
        """
        from megatron.core.optimizer.optimizer import ChainedOptimizer
        optimizers = optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else [optimizer]
        for _optimizer in optimizers:
            # Megatron wraps a plain torch optimizer; fall back to the object itself so a bare one
            # also works. `optimizer` being None means this rank owns no parameters, which is
            # legitimate under a custom pipeline layout.
            inner = getattr(_optimizer, 'optimizer', _optimizer)
            if inner is None or getattr(_optimizer, 'is_stub_optimizer', False):
                continue
            # The inner param_groups hold the optimizer's own handle on each parameter. For
            # bf16/fp16 training under a DistributedOptimizer these are separate fp32 master shards
            # (``shard_fp32_from_float16_groups``, allocated with ``.clone().float()``): they own
            # their storage, so offloading only the DDP param/grad buffers would leave the master
            # weights (and below, the momentum) resident -- hence this walk.
            #
            # For FP32 training the DistributedOptimizer does NOT clone. It aliases its inner params
            # to VIEWS of the DDP flat param buffer (``_build_model_and_main_param_groups`` slices
            # ``model_param.view(-1)[start:end]`` straight into ``shard_fp32_params``). The
            # ``buffer.offload_to_cpu()`` above already copied that buffer to pinned host memory and
            # freed its device storage (``param_data.storage().resize_(0)``), so those views now
            # dangle and a D2H copy of one raises CUDA "invalid argument". Their bytes are safe in
            # the buffer's host copy and reappear on ``reload_from_cpu()``, so skip exactly the params
            # whose storage was freed and move only the ones still owning live storage (the
            # bf16/fp16 master shards). Skipping is required for correctness, not just to dodge the
            # crash: reassigning a freed view's ``.data`` to a fresh host tensor would detach it from
            # the buffer the model itself reads, so ``reload_from_cpu()`` would restore the model's
            # weights while the optimizer kept stepping on an orphaned copy.
            for group in inner.param_groups:
                for param in group['params']:
                    if isinstance(param, torch.Tensor) and param.untyped_storage().size() > 0:
                        param.data = param.data.to(device, non_blocking=True)
            # Momentum/variance (exp_avg, exp_avg_sq, ...) are always fresh allocations, never buffer
            # views, so the freed-storage guard is a no-op here; it just keeps the walk uniform.
            for state in inner.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.untyped_storage().size() > 0:
                        state[key] = value.to(device, non_blocking=True)
