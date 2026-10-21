# Copyright (c) ModelScope Contributors. All rights reserved.
"""GPU/NPU 通用的 delta 同步脚本辅助函数（delta_validation.py / delta_benchmark.py 共用）。

这里的大部分函数是纯 Python（不启动 Ray、不接触设备），因此可以在没有加速卡的机器上做 --dry-run 和单测。
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_NPU_CARDS = 4

# vLLM-Ascend 在 RL 场景下必须带 rl_config（它会关闭 FRACTAL_NZ，否则 start_weight_update / wake_up 报 weight_nz_mode 错误）。
NPU_ADDITIONAL_CONFIG = {'rl_config': {'enabled': True}}

TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'tokenizer.model', 'vocab.json', 'merges.txt')

OFFLINE_ENV = {
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_DATASETS_OFFLINE': '1',
    'MODELSCOPE_OFFLINE': '1',  # 尽力而为：不同 modelscope 版本的离线开关不一致，真正的保护是本地目录检查
}

NPU_REQUIREMENTS = (
    "vLLM-Ascend 需要 additional_config={'rl_config': {'enabled': True}}（脚本已自动加入）；否则 "
    'start_weight_update/wake_up 会抛 weight_nz_mode 的 ValueError。',
    '分离部署（trainer 与 rollout 使用不同的卡）通常不需要 sleep mode：NPU 默认 free_cache_engine=False；'
    '若用 --free-cache-engine 打开，需要从源码编译的 vllm-ascend。',
    "sampler 的 engine_args 必须带 weight_transfer_config={'backend': 'twinkle_delta_hccl'}（CUDA 为 twinkle_delta_ipc）。",
    'vLLM 必须提供 engine-owned WeightTransferEngine API（vLLM #44353）和 checkpoint patch API（vLLM #50723）；'
    '可先运行 cookbook/checkpoint_engine/check_npu_env.py 检查。',
    '支持范围：仅 vLLM、BF16 稠密（非 MoE）、非量化、无推测解码、vLLM 内部 data_parallel_size=1、'
    'trainer FSDP2 且 TP=PP=EP=1、无 LoRA、不支持 rebuild_group；rollout 可有多个副本（replica 为主、tp 为次）。',
    'GPU 与 NPU 的其余配置接口一致。以上内容尚未在真实 NPU 上验证。',
)


@dataclass(frozen=True)
class PlatformSpec:
    name: str  # twinkle Platform 名称，同时用作 DeviceGroup.device_type 与 CheckpointEngineManager(platform=)
    torch_device: str  # torch 设备模块名：torch.cuda / torch.npu
    backend: str  # vLLM weight_transfer_config 的 backend


PLATFORM_SPECS = {
    'GPU': PlatformSpec('GPU', 'cuda', 'twinkle_delta_ipc'),
    'NPU': PlatformSpec('NPU', 'npu', 'twinkle_delta_hccl'),
}
_PLATFORM_ALIASES = {'GPU': 'GPU', 'CUDA': 'GPU', 'NPU': 'NPU', 'ASCEND': 'NPU'}


def normalize_platform(value: str) -> str:
    """把 gpu/cuda/npu/ascend（不区分大小写）归一化为 'GPU' 或 'NPU'。"""
    try:
        return _PLATFORM_ALIASES[str(value).upper()]
    except KeyError:
        raise argparse.ArgumentTypeError(f'未知平台 {value!r}，可选 GPU 或 NPU') from None


def add_platform_arguments(parser: argparse.ArgumentParser) -> None:
    """两个脚本共用的 GPU/NPU、离线模型与 dry-run 参数。"""
    parser.add_argument('--platform', type=normalize_platform, default='GPU', metavar='{GPU,NPU}',
                        help='设备平台（默认 GPU；cuda 是 GPU 的别名）')
    parser.add_argument('--model-path', help='本地模型目录（不联网）；给出后覆盖 --model')
    parser.add_argument('--offline', action=argparse.BooleanOptionalAction, default=None,
                        help='禁止任何联网下载（设置 HF_HUB_OFFLINE 等）；NPU 默认开启，GPU 默认关闭')
    parser.add_argument('--num-devices', type=int,
                        help='可用卡数，用于预检卡数预算；NPU 默认 4，GPU 默认不限制')
    parser.add_argument('--free-cache-engine', action=argparse.BooleanOptionalAction, default=None,
                        help='DeltaSyncConfig.free_cache_engine；GPU 默认开启，NPU 默认关闭（需要 sleep mode）')
    parser.add_argument('--use-token-ids', action=argparse.BooleanOptionalAction, default=None,
                        help='让 vLLM 使用 skip_tokenizer_init，只用 token id 生成并比较；模型目录没有 tokenizer 时自动开启')
    parser.add_argument('--dry-run', action='store_true',
                        help='只解析参数、构造 DeviceGroup/配置并做卡数预检，打印执行计划后退出（不启动 Ray、不接触设备）')


def has_tokenizer(model_dir: str) -> bool:
    return any((Path(model_dir) / name).is_file() for name in TOKENIZER_FILES)


def resolve_model(args) -> None:
    """解析 --model-path / --offline / --use-token-ids，原地更新 args。"""
    if args.model_path:
        args.model = args.model_path
    if args.offline is None:
        args.offline = args.platform == 'NPU'
    if args.num_devices is None and args.platform == 'NPU':
        args.num_devices = DEFAULT_NPU_CARDS
    if args.free_cache_engine is None:
        args.free_cache_engine = args.platform == 'GPU'
    if args.use_token_ids is None:
        local = Path(args.model).is_dir()
        args.use_token_ids = bool(local and not has_tokenizer(args.model))


def apply_offline_env(args) -> dict[str, str]:
    """--offline 时设置离线环境变量（必须在启动 Ray 之前调用，Ray 子进程会继承）。"""
    if not args.offline:
        return {}
    for key, value in OFFLINE_ENV.items():
        os.environ[key] = value
    return dict(OFFLINE_ENV)


def preflight(args) -> tuple[list[str], list[str]]:
    """返回 (errors, warnings)。不接触设备。"""
    errors: list[str] = []
    warnings: list[str] = []
    trainer, dp, tp = args.trainer_gpus, args.dp, args.tp
    if dp < 1 or tp < 1 or trainer < 1:
        errors.append('--trainer-gpus、--dp、--tp 必须为正数')
        return errors, warnings
    needed = trainer + dp * tp
    if args.num_devices is not None and needed > args.num_devices:
        errors.append(
            f'卡数预算不足：trainer {trainer} + rollout dp*tp={dp}*{tp}={dp * tp} = {needed} 张，'
            f'超过 --num-devices={args.num_devices}。4 卡可验证的布局：'
            '(a) --trainer-gpus 1 --dp 1 --tp 1；(b) --trainer-gpus 1 --dp 1 --tp 2；'
            '(c) --trainer-gpus 2 --dp 2 --tp 1；(d) --trainer-gpus 2 --dp 1 --tp 2')
    if args.platform == 'NPU' and dp > 1 and tp > 1 and (args.num_devices or DEFAULT_NPU_CARDS) <= DEFAULT_NPU_CARDS:
        warnings.append(f'rollout 副本数>1 且 tp>1（dp={dp}, tp={tp}）至少需要 {1 + dp * tp} 张卡，'
                        f'4 卡机器无法验证“多副本 + tp>1”')
    model_dir = Path(args.model)
    if args.offline and not model_dir.is_dir():
        errors.append(f'--offline 下模型必须是本地目录（--model-path），当前为 {args.model!r}')
    if model_dir.is_dir():
        if not (model_dir / 'config.json').is_file():
            errors.append(f'模型目录 {args.model} 缺少 config.json')
        if not has_tokenizer(args.model) and not args.use_token_ids:
            errors.append(f'模型目录 {args.model} 没有 tokenizer 文件，请加 --use-token-ids')
    if args.platform == 'NPU' and args.free_cache_engine:
        warnings.append('NPU 上启用 free_cache_engine 需要 sleep mode（已加入 enable_sleep_mode=True），'
                        '且需要从源码编译的 vllm-ascend')
    if args.platform == 'NPU' and args.strategy == 'accelerate' and args.cpu_offload:
        warnings.append('NPU 上的 Accelerate FSDP2 CPU offload 未验证')
    return errors, warnings


def build_engine_args(args, worker_extension_cls: str) -> dict[str, Any]:
    spec = PLATFORM_SPECS[args.platform]
    engine_args: dict[str, Any] = dict(
        tensor_parallel_size=args.tp, dtype='bfloat16', max_model_len=256, enforce_eager=True,
        gpu_memory_utilization=0.4, worker_extension_cls=worker_extension_cls,
        weight_transfer_config={'backend': spec.backend})
    if args.platform == 'NPU':
        engine_args['additional_config'] = {'rl_config': dict(NPU_ADDITIONAL_CONFIG['rl_config'])}
        if args.free_cache_engine:
            engine_args['enable_sleep_mode'] = True
    if args.use_token_ids:
        engine_args['skip_tokenizer_init'] = True
    return engine_args


def delta_config_kwargs(args) -> dict[str, Any]:
    return dict(bucket_bytes=args.bucket_bytes, gather_batch_size=args.gather_batch_size,
                free_cache_engine=bool(args.free_cache_engine))


def build_device_groups(args) -> list:
    from twinkle import DeviceGroup
    spec = PLATFORM_SPECS[args.platform]
    trainer, rollout = args.trainer_gpus, args.dp * args.tp
    return [
        DeviceGroup(name='trainer', ranks=list(range(trainer)), device_type=spec.name),
        DeviceGroup(name='rollout', ranks=list(range(trainer, trainer + rollout)), device_type=spec.name,
                    gpus_per_worker=args.tp),
    ]


def nproc_per_node(args) -> int:
    return args.devices_per_node or args.trainer_gpus + args.dp * args.tp


def print_plan(args, worker_extension_cls: str) -> None:
    """打印解析后的执行计划（--dry-run 以及真实运行前都会打印）。"""
    from twinkle.checkpoint_engine import DeltaSyncConfig
    spec = PLATFORM_SPECS[args.platform]
    print(f'== 执行计划 platform={args.platform} (torch.{spec.torch_device}) ==')
    print(f'model          : {args.model}  (offline={args.offline}, use_token_ids={args.use_token_ids})')
    print(f'cards          : trainer={args.trainer_gpus} + rollout dp*tp={args.dp}*{args.tp}='
          f'{args.dp * args.tp}  / num_devices={args.num_devices}')
    for group in build_device_groups(args):
        print(f'device group   : {group}')
    print(f'nproc_per_node : {nproc_per_node(args)}')
    print(f'manager        : CheckpointEngineManager(platform={args.platform!r}, mode=\'standalone\', '
          f"sync_backend='delta_sharded')")
    print(f'backend        : {spec.backend}')
    print(f'engine_args    : {build_engine_args(args, worker_extension_cls)}')
    print(f'DeltaSyncConfig: {DeltaSyncConfig(**delta_config_kwargs(args))}')
    if args.offline:
        print(f'offline env    : {OFFLINE_ENV}')


def run_preflight_or_exit(args, parser: argparse.ArgumentParser, worker_extension_cls: str) -> None:
    """解析 -> 预检 -> 打印计划；--dry-run 时在此退出（返回前不接触设备）。"""
    resolve_model(args)
    errors, warnings = preflight(args)
    if args.platform == 'NPU':
        print('== NPU 使用须知（均未在真实 NPU 上验证）==')
        for line in NPU_REQUIREMENTS:
            print(f'  - {line}')
    for warning in warnings:
        print(f'[警告] {warning}')
    if errors:
        parser.error('预检失败：\n  ' + '\n  '.join(errors))
    print_plan(args, worker_extension_cls)
    if args.dry_run:
        print('--dry-run：未启动 Ray，未接触设备，退出。')
        raise SystemExit(0)
    apply_offline_env(args)


def torch_device_module(platform_or_type: str):
    """返回 torch.cuda 或 torch.npu（按 'GPU'/'NPU'/'cuda'/'npu' 选择）；NPU 会先 import torch_npu。"""
    import torch
    name = platform_or_type.lower()
    name = PLATFORM_SPECS[normalize_platform(name)].torch_device if name in ('gpu', 'npu', 'ascend') else name
    if name == 'npu':
        import torch_npu  # noqa: F401
    return getattr(torch, name)


def device_peak_bytes(device_type: str) -> int:
    return int(torch_device_module(device_type).max_memory_allocated())


def device_synchronize(device_type: str) -> None:
    torch_device_module(device_type).synchronize()
