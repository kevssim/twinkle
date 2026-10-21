# Copyright (c) ModelScope Contributors. All rights reserved.
"""在 GPU(CUDA) 或 Ascend NPU 上验证 FSDP2 增量同步：逐字节比较 delta 与 full 同步后的 vLLM 权重。

示例（2 张训练卡 + 4 张 rollout 卡，GPU）：
python -m cookbook.checkpoint_engine.delta_validation --dp 2 --tp 2 --rounds 10
示例（只有 2 张卡：1 张训练、1 张 rollout，FSDP 不分片）：
python -m cookbook.checkpoint_engine.delta_validation --trainer-gpus 1 --dp 1 --tp 1 --rounds 2

NPU（单机 4 卡、不联网，模型必须是本地目录；NPU 默认 --offline 且 --num-devices 4）。先用 --dry-run 检查计划：
(a) 1 训练 + 1 rollout(tp=1)：
python -m cookbook.checkpoint_engine.delta_validation --platform NPU --model-path /path/to/model \\
    --trainer-gpus 1 --dp 1 --tp 1 --rounds 2
(b) 1 训练 + 1 rollout(tp=2)：  --trainer-gpus 1 --dp 1 --tp 2
(c) 2 训练 + 2 个 rollout 副本(tp=1)：  --trainer-gpus 2 --dp 2 --tp 1
(d) 2 训练 + 1 rollout(tp=2)：  --trainer-gpus 2 --dp 1 --tp 2
“多副本且 tp>1”至少需要 5 张卡，4 卡机器无法验证。NPU 路径尚未在真实 NPU 上跑过。
"""
import argparse
import importlib.metadata
import json
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
from torch.distributed.tensor import DTensor

import twinkle
from cookbook.checkpoint_engine.device_utils import (add_platform_arguments, build_device_groups, build_engine_args,
                                                      delta_config_kwargs, device_peak_bytes, device_synchronize,
                                                      nproc_per_node, run_preflight_or_exit)
from twinkle import DeviceMesh, Platform, remote_class, remote_function
from twinkle.checkpoint_engine import CheckpointEngineManager, DeltaSyncConfig
from twinkle.checkpoint_engine.delta.spec import ShardSpec, derive_dtensor_placement, translate_flat_indices
from twinkle.data_format.sampling import SamplingParams
from twinkle.model import TransformersModel
from twinkle.sampler import vLLMSampler
from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension

# tp>1 时 vLLM 会初始化 NCCL 通信组；在 HAMi 等 vGPU 环境下，NCCL 的 cuMem* 路径会触发
# "Illegal device id" 并 segfault。与 verl 一致，在 Ray actor 继承环境前关闭（可由外部覆盖）。
os.environ.setdefault('NCCL_CUMEM_ENABLE', '0')
os.environ.setdefault('NCCL_CUMEM_HOST_ENABLE', '0')

# 0% 用来测固定开销；其余贴近真实 RL 单步变化（稠密模型约 1%~3%）。
CHANGE_RATIOS = (0., .005, .01, .02, .03)


def log_progress(message: str) -> None:
    print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}', flush=True)


@contextmanager
def validation_stage(name: str):
    log_progress(f'开始：{name}')
    started = perf_counter()
    try:
        yield
    except Exception as error:
        log_progress(f'失败：{name}，耗时 {perf_counter() - started:.3f}s，{error!r}')
        raise
    else:
        log_progress(f'完成：{name}，耗时 {perf_counter() - started:.3f}s')


def environment_report():
    """记录软件版本和驱动信息，方便复现设备上的结果。"""
    report = {}
    for package in ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'transformers', 'accelerate', 'ray'):
        try:
            report[package] = importlib.metadata.version(package.replace('_', '-'))
        except importlib.metadata.PackageNotFoundError:
            report[package] = None
    # GPU 机器有 nvidia-smi，Ascend 机器有 npu-smi；哪个存在就记录哪个。
    for tool, command in (('nvidia-smi', ['nvidia-smi']), ('npu-smi', ['npu-smi', 'info'])):
        try:
            report[tool] = subprocess.run(command, capture_output=True, text=True, timeout=15).stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            report[tool] = None
    return report


@remote_class()
class ValidationTrainer(TransformersModel):
    def __init__(self, device_mesh: DeviceMesh = None, **kwargs):
        TransformersModel.__init__.__wrapped__(self, device_mesh=device_mesh, **kwargs)

    @remote_function(dispatch='all', lazy_collect=False)
    def change_fraction(self, ratio: float):
        """翻转每个参数前 ratio 比例元素的符号位，产生可控的 BF16 字节变化。"""
        self._lazy_wrap_model()
        with torch.no_grad():
            for tensor in self.strategy.unwrap_model(self.model).parameters():
                place, _, _ = derive_dtensor_placement(ShardSpec.from_param(tensor))
                local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
                values = local.detach().to(torch.bfloat16).contiguous().reshape(-1)
                indices = torch.arange(values.numel(), dtype=torch.int64, device=values.device)
                # 使用完整参数的坐标，使各 FSDP shard 合起来恰好修改指定比例。
                changed = translate_flat_indices(indices, place) < int(tensor.numel() * ratio)
                bits = values.view(torch.int16)
                bits[changed] = bits[changed].bitwise_xor(-32768)
                local.copy_(values.reshape(local.shape).to(device=local.device, dtype=local.dtype))

    @remote_function(dispatch='all', lazy_collect=False)
    def forbid_full_tensor(self):
        # 在 trainer 进程里临时禁用全量聚合，检查稳态 delta 没有退回 full 路径。
        self._validation_full_tensor = DTensor.full_tensor

        def forbidden(*args, **kwargs):
            raise AssertionError('稳态 delta 调用了 DTensor.full_tensor()')

        DTensor.full_tensor = forbidden

    @remote_function(dispatch='all', lazy_collect=False)
    def allow_full_tensor(self):
        DTensor.full_tensor = self._validation_full_tensor

    @remote_function(dispatch='all', lazy_collect=False)
    def training_step(self):
        """执行一次真实的前向、反向和 SGD 更新，补充人为修改之外的验证。"""
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        tokens = torch.tensor([[1, 2, 3, 4]], device=self._delta_device)
        self.model.train()
        optimizer.zero_grad()
        loss = self.model(input_ids=tokens, labels=tokens).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return float(loss.detach().cpu())

    @remote_function(dispatch='all', lazy_collect=False)
    def sync_device(self):
        """等待本进程设备上已排队的 kernel 全部完成。

        端到端耗时基准用它隔开“训练计算”和“权重同步”：clip_grad_and_step 返回时优化器 kernel 可能还没跑完，
        不同步的话这部分时间会被算进随后的 sync。
        """
        device_synchronize(Platform.get_platform().device_prefix())

    @remote_function(dispatch='all', lazy_collect=False)
    def validation_environment(self):
        return environment_report()


class ValidationWorker(TwinkleWorkerExtension):
    """在每个 vLLM TP worker 内保存和比较真正参与推理的 runtime 权重。"""

    def validation_environment(self):
        return environment_report()

    def capture_weights(self):
        self._validation_weights = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model_runner.model.state_dict().items()}
        # self.device 是 torch.device('cuda:N') 或 torch.device('npu:N')，不要写死 torch.cuda。
        return {'device_peak_bytes': device_peak_bytes(torch.device(self.device).type)}

    def begin_full_reload(self):
        # vLLM 可能打包或转置权重；全量基准也必须走完整 reload 生命周期。
        from vllm.model_executor.model_loader.reload.layerwise import (
            initialize_layerwise_reload, record_metadata_for_reloading)
        with torch.device(self.device):
            record_metadata_for_reloading(self.model_runner.model)
            initialize_layerwise_reload(self.model_runner.model)

    def compare_weights(self):
        from vllm.model_executor.model_loader.reload.layerwise import finalize_layerwise_reload
        finalize_layerwise_reload(self.model_runner.model, self.model_runner.model_config)
        state = self.model_runner.model.state_dict()
        if state.keys() != self._validation_weights.keys():
            raise AssertionError('全量 reload 改变了 runtime 权重键')
        mismatches = []
        for name, tensor in state.items():
            # 比较字节而非浮点近似值，包含正负零等位级差异。
            actual = tensor.detach().cpu().contiguous()
            expected = self._validation_weights[name].contiguous()
            if actual.shape != expected.shape or not torch.equal(
                    actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)):
                mismatches.append(self._describe_mismatch(name, actual, expected))
        self._validation_weights = None
        if mismatches:
            # 汇总全部不一致的 tensor，而不是在第一个处中止，便于判断是整体错误还是局部错误。
            shown = '\n  '.join(mismatches[:20])
            more = f'\n  ...另有 {len(mismatches) - 20} 个' if len(mismatches) > 20 else ''
            raise AssertionError(
                f'delta 与 full 权重不一致：rank={self.rank}, {len(mismatches)}/{len(state)} 个 tensor 不一致\n'
                f'  {shown}{more}')
        return True

    @staticmethod
    def _describe_mismatch(name, actual, expected):
        """描述一个不一致 tensor：不一致元素占比、行范围，以及首个差异处的 full/delta 取值。

        actual 是 full reload 后的运行时权重，expected 是 delta 同步后保存的快照。
        """
        if actual.shape != expected.shape:
            return f'{name}: shape 不同 full={tuple(actual.shape)} delta={tuple(expected.shape)}'
        int_dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[actual.element_size()]
        differs = actual.reshape(-1).view(int_dtype) != expected.reshape(-1).view(int_dtype)
        total = differs.numel()
        count = int(differs.sum())
        first = int(differs.nonzero()[0])
        last = int(differs.nonzero()[-1])
        text = (f'{name}: shape={tuple(actual.shape)} 不一致元素 {count}/{total} ({count / total:.4%}), '
                f'flat 范围 [{first}, {last}]')
        if actual.dim() >= 2:
            rows = differs.reshape(actual.shape[0], -1).any(dim=1).nonzero().flatten()
            text += (f', 不一致行 {rows.numel()}/{actual.shape[0]} '
                     f'(行范围 [{int(rows[0])}, {int(rows[-1])}])')
        full_value = actual.reshape(-1)[first].item()
        delta_value = expected.reshape(-1)[first].item()
        return f'{text}, 首个差异 flat={first}: full={full_value} delta={delta_value}'


@remote_class()
class ValidationSampler(vLLMSampler):
    def __init__(self, device_mesh: DeviceMesh = None, **kwargs):
        vLLMSampler.__init__.__wrapped__(self, device_mesh=device_mesh, **kwargs)

    @remote_function(dispatch='all', lazy_collect=False, collect='flatten')
    def worker_rpc(self, method: str):
        # 遍历全部 DP 副本及其 TP workers，任何 worker 的比较失败都会向上传递。
        return self._run_in_loop(self.engine.engine.collective_rpc(method))

    @remote_function(dispatch='all', lazy_collect=False)
    def pause_validation(self):
        self._run_in_loop(self.engine.pause_delta_generation(self._delta_config.timeout_seconds))

    @remote_function(dispatch='all', lazy_collect=False)
    def generate_one_token(self):
        return self._run_in_loop(self.engine.sample(
            prompt=[1, 2, 3], sampling_params=SamplingParams(max_tokens=1, temperature=0.)))


WORKER_EXTENSION_CLS = 'cookbook.checkpoint_engine.delta_validation.ValidationWorker'


def build_components(args):
    """训练使用 trainer_gpus 张卡的 FSDP2；rollout 使用 dp 个独立副本，每个副本占 tp 张卡。

    GPU 与 NPU 共用同一套流程，差别（device_type、vLLM 的 weight_transfer 后端、NPU 的 rl_config 等）
    都在 device_utils 里按 --platform 解析。
    """
    trainer_gpus = args.trainer_gpus
    rollout_gpus = args.dp * args.tp
    with validation_stage(f'初始化 Ray 和 {args.platform} 资源组'):
        twinkle.initialize(mode='ray', nproc_per_node=nproc_per_node(args),
                           groups=build_device_groups(args), lazy_collect=False)
    with validation_stage(f'创建 {trainer_gpus} 卡 FSDP trainer'):
        trainer = ValidationTrainer(
            model_id=args.model, remote_group='trainer', strategy=args.strategy, mixed_precision='bf16',
            fsdp_config={'cpu_offload': True} if args.cpu_offload else {},
            device_mesh=DeviceMesh.from_sizes(world_size=trainer_gpus, fsdp_size=trainer_gpus))
    with validation_stage(f'创建 rollout：{args.dp} 个副本，每个 {args.tp} 张 GPU'):
        sampler = ValidationSampler(
            model_id=args.model, remote_group='rollout',
            device_mesh=DeviceMesh.from_sizes(world_size=rollout_gpus, tp_size=args.tp, dp_size=args.dp),
            engine_args=build_engine_args(args, WORKER_EXTENSION_CLS))
    return trainer, sampler


def steady_sync(trainer, delta):
    """稳态同步必须只传增量；无论是否成功，都恢复 full_tensor 方法。"""
    with validation_stage('禁用 trainer 的全量权重聚合'):
        trainer.forbid_full_tensor()
    try:
        started = perf_counter()
        with validation_stage('稳态 delta 权重同步'):
            delta.sync_weights()
        if delta.last_delta_metrics['seed']:
            raise AssertionError('稳态同步意外重新执行了全量 seed')
        return perf_counter() - started
    finally:
        with validation_stage('恢复 trainer 的全量权重聚合'):
            trainer.allow_full_tensor()


def compare_with_full(sampler, full):
    """保存 delta 结果，再全量加载相同 trainer 权重，比较两条路径的最终结果。"""
    with validation_stage('rollout 生成一个 token，检查同步后推理'):
        sampler.generate_one_token()
    with validation_stage('暂停 rollout 推理'):
        sampler.pause_validation()
    with validation_stage('保存所有 rollout worker 的 delta 权重快照'):
        workers = sampler.worker_rpc('capture_weights')
    with validation_stage('准备全量权重 reload'):
        sampler.worker_rpc('begin_full_reload')
    started = perf_counter()
    with validation_stage('full 权重同步'):
        full.sync_weights()
    seconds = perf_counter() - started
    with validation_stage('所有 rollout worker 逐字节比较 delta 与 full 权重'):
        sampler.worker_rpc('compare_weights')
    return {'full_seconds': seconds, 'workers': workers}


def run_validation(args):
    log_progress(
        f'启动验证：platform={args.platform}，model={args.model}，strategy={args.strategy}，'
        f'trainer={args.trainer_gpus} 张卡，rollout dp={args.dp}、tp={args.tp}，'
        f'rounds={args.rounds}，report={Path(args.report).resolve()}')
    trainer, sampler = build_components(args)
    with validation_stage('创建 delta 和 full checkpoint manager'):
        delta = CheckpointEngineManager(
            trainer, sampler, platform=args.platform, mode='standalone', sync_backend='delta_sharded',
            delta_config=DeltaSyncConfig(**delta_config_kwargs(args)))
        full = CheckpointEngineManager(trainer, sampler, platform=args.platform, mode='standalone')
    report = {'status': 'running', 'arguments': vars(args), 'rounds': []}

    def save_report():
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    def check_round(stage):
        with validation_stage(stage):
            seconds = steady_sync(trainer, delta)
            result = compare_with_full(sampler, full)
        result.update(stage=stage, delta_seconds=seconds, metrics=delta.last_delta_metrics)
        report['rounds'].append(result)
        save_report()
        log_progress(f'{stage}：字节比较通过，delta={seconds:.3f}s，full={result["full_seconds"]:.3f}s')

    try:
        with validation_stage('收集 driver 和 trainer 环境信息'):
            report.update(driver=environment_report(), trainers=trainer.validation_environment())
        # 1. 首次发送 dense seed 并建立 CPU shard 快照，后续各轮复用同一个通信组。
        with validation_stage('首次全量 seed 同步，建立增量快照和通信组'):
            delta.sync_weights()
        with validation_stage('收集 rollout worker 环境信息'):
            report['workers'] = sampler.worker_rpc('validation_environment')
        with validation_stage('首次 seed 权重验证'):
            report['seed'] = compare_with_full(sampler, full)

        # 2. 轮流覆盖零变化、稀疏变化和全部变化；full 基准不修改 trainer 快照。
        for step in range(args.rounds):
            ratio = CHANGE_RATIOS[step % len(CHANGE_RATIOS)]
            with validation_stage(f'第 {step + 1}/{args.rounds} 轮：修改 {ratio:.1%} 权重'):
                trainer.change_fraction(ratio)
            check_round(f'第 {step + 1}/{args.rounds} 轮验证，修改比例 {ratio:.1%}')

        # 3. 真实训练更新后的 delta 仍必须与 full 逐字节一致。
        with validation_stage('真实训练：前向、反向和 SGD 更新'):
            report['training_loss'] = trainer.training_step()
        log_progress(f'训练 loss={report["training_loss"]}')
        check_round('真实训练更新')

        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=repr(error))
        log_progress(f'验证失败：{error!r}')
        raise
    finally:
        # 4. 关闭接收 actors、通信组和快照；即使验证失败也保存报告。
        try:
            with validation_stage('关闭增量同步会话，释放通信组和快照'):
                delta.close_sync_session()
        except Exception as error:
            report.update(status='failed', cleanup_error=repr(error))
            raise
        finally:
            with validation_stage(f'保存验证报告：{args.report}'):
                save_report()
            log_progress(f'验证结束：status={report["status"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_platform_arguments(parser)  # --platform {GPU,NPU}、--model-path、--offline、--num-devices、--dry-run 等
    parser.add_argument('--model', default='Qwen/Qwen3-0.6B', help='训练与 rollout 使用的同一模型')
    parser.add_argument('--trainer-gpus', type=int, choices=(1, 2), default=2, help='FSDP2 训练卡数；1 表示不分片')
    parser.add_argument('--tp', type=int, choices=(1, 2), default=1, help='每个 rollout 副本的卡数')
    parser.add_argument('--dp', type=int, default=1, help='独立 rollout 副本数')
    parser.add_argument('--strategy', choices=('native_fsdp', 'accelerate'), default='native_fsdp')
    parser.add_argument('--cpu-offload', action='store_true', help='使用 Accelerate FSDP2 CPU offload')
    parser.add_argument('--rounds', type=int, default=1, help='不同变化比例的验证轮数')
    parser.add_argument('--gather-batch-size', type=int, default=32)
    parser.add_argument('--bucket-bytes', type=int, default=64 << 20)
    parser.add_argument('--devices-per-node', type=int, help='每个节点的卡数，默认单节点运行')
    parser.add_argument('--report', default='delta-validation.json', help='结果报告路径')
    args = parser.parse_args()
    if args.dp < 1 or args.rounds < 1:
        parser.error('--dp 和 --rounds 必须为正数')
    if args.cpu_offload and args.strategy != 'accelerate':
        parser.error('--cpu-offload 需要 --strategy accelerate')
    # 解析模型/离线设置、预检卡数预算并打印计划；--dry-run 在这里退出，不启动 Ray。
    run_preflight_or_exit(args, parser, WORKER_EXTENSION_CLS)
    run_validation(args)


if __name__ == '__main__':
    main()
