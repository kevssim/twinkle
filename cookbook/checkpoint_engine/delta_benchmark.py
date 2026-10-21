# Copyright (c) ModelScope Contributors. All rights reserved.
"""权重同步耗时基准，口径对齐 verl 的 tests/checkpoint_engine/bench_delta_vllm.py。

和 delta_validation.py 不同，这里不做逐字节比较，只计时 manager.sync_weights()：
先做一次 seed 同步，再对每个变化比例各同步 --repeats 次。一个进程只测一个后端。

python -m cookbook.checkpoint_engine.delta_benchmark --model /nas/disk1/Qwen3-8B --backend delta_sharded
python -m cookbook.checkpoint_engine.delta_benchmark --model /nas/disk1/Qwen3-8B --backend full
"""
import argparse
from statistics import median
from time import perf_counter

from cookbook.checkpoint_engine.delta_validation import build_components, log_progress
from twinkle.checkpoint_engine import CheckpointEngineManager, DeltaSyncConfig


def timed_sync(manager) -> float:
    started = perf_counter()
    manager.sync_weights()
    return perf_counter() - started


def run_benchmark(args):
    trainer, sampler = build_components(args)
    if args.backend == 'delta_sharded':
        manager = CheckpointEngineManager(
            trainer, sampler, platform='GPU', mode='standalone', sync_backend='delta_sharded',
            delta_config=DeltaSyncConfig(bucket_bytes=args.bucket_bytes, gather_batch_size=args.gather_batch_size))
    else:
        manager = CheckpointEngineManager(trainer, sampler, platform='GPU', mode='standalone')

    def metrics():
        return manager.last_delta_metrics if args.backend == 'delta_sharded' else {}

    try:
        seconds = timed_sync(manager)
        log_progress(f'[seed] backend={args.backend} {seconds:.3f}s metrics={metrics()}')

        rows = []
        step = 0
        for ratio in args.ratios:
            samples = []
            for _ in range(args.repeats):
                step += 1
                trainer.change_fraction(ratio)
                seconds = timed_sync(manager)
                samples.append(seconds)
                log_progress(f'[ratio={ratio}] step={step} sync={seconds:.3f}s metrics={metrics()}')
            rows.append((ratio, samples, metrics()))
    finally:
        if args.backend == 'delta_sharded':
            manager.close_sync_session()

    print(f'\nbackend={args.backend} model={args.model} trainer={args.trainer_gpus} rollout_tp={args.tp} dp={args.dp} '
          f'bucket={args.bucket_bytes >> 20}MB')
    print(f"{'ratio':>8} {'median(s)':>10} {'min(s)':>8} {'max(s)':>8}  last metrics")
    for ratio, samples, last in rows:
        print(f'{ratio:>8} {median(samples):>10.3f} {min(samples):>8.3f} {max(samples):>8.3f}  {last}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='训练与 rollout 使用的同一模型，例如 Qwen3-8B')
    parser.add_argument('--backend', choices=('delta_sharded', 'full'), default='delta_sharded')
    parser.add_argument('--trainer-gpus', type=int, choices=(1, 2), default=2, help='FSDP2 训练卡数')
    parser.add_argument('--tp', type=int, choices=(1, 2), default=2, help='rollout 副本的 GPU 数')
    parser.add_argument('--dp', type=int, default=1, help='独立 rollout 副本数')
    parser.add_argument('--strategy', choices=('native_fsdp', 'accelerate'), default='native_fsdp')
    parser.add_argument('--cpu-offload', action='store_true', help='使用 Accelerate FSDP2 CPU offload')
    parser.add_argument('--ratios', type=float, nargs='+', default=[0.005, 0.01, 0.02, 0.03])
    parser.add_argument('--repeats', type=int, default=2, help='每个比例的同步次数')
    parser.add_argument('--bucket-bytes', type=int, default=2048 << 20, help='verl 默认 2048MB')
    parser.add_argument('--gather-batch-size', type=int, default=32)
    parser.add_argument('--devices-per-node', type=int, help='每个节点的 GPU 数，默认单节点运行')
    args = parser.parse_args()
    if args.cpu_offload and args.strategy != 'accelerate':
        parser.error('--cpu-offload 需要 --strategy accelerate')
    run_benchmark(args)


if __name__ == '__main__':
    main()
