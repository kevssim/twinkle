# Copyright (c) ModelScope Contributors. All rights reserved.
"""端到端“一个训练 step”耗时对比：delta_sharded vs full。

一个 step = 训练计算（前向 + 反向 + AdamW，forward_backward + clip_grad_and_step）+ 权重同步（manager.sync_weights，阻塞到 rollout 就绪）。
一个进程只测一个后端，各自写 JSON 报告，最后用 --compare 合并成对比表（不启动 Ray）。

  # 1. 分别跑两个后端（两次之间重启 Ray 集群/actors，保证干净）
  python -m cookbook.checkpoint_engine.e2e_step_benchmark --platform NPU --model-path /nas/disk1/Qwen3.5-9B \\
      --backend delta_sharded --trainer-gpus 2 --dp 1 --tp 2 --devices-per-node 2 \\
      --bucket-bytes 2147483648 --seq-len 128 --report e2e-delta.json
  python -m cookbook.checkpoint_engine.e2e_step_benchmark --platform NPU --model-path /nas/disk1/Qwen3.5-9B \\
      --backend full --trainer-gpus 2 --dp 1 --tp 2 --devices-per-node 2 \\
      --bucket-bytes 2147483648 --seq-len 128 --report e2e-full.json
  # 2. 合并对比
  python -m cookbook.checkpoint_engine.e2e_step_benchmark --compare e2e-delta.json e2e-full.json --md e2e.md

默认只测真实 step：训练一步（前向 + 反向 + AdamW）后立刻 sync，变化比例由 AdamW 自然产生
（报告里记录 delta 的实际 changed_ratio）。首次 seed 同步和 --warmup-steps 指定的预热步只记录、不进统计。
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import median
from time import perf_counter

from cookbook.checkpoint_engine.device_utils import add_platform_arguments, delta_config_kwargs, run_preflight_or_exit


def log_progress(message: str) -> None:
    print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}', flush=True)


def delta_info(manager) -> dict:
    """从 manager.last_delta_metrics 里取 delta 的实际变化比例和 payload；full 后端返回 {}。"""
    metrics = getattr(manager, 'last_delta_metrics', None)
    if not metrics:
        return {}
    info = {'manager_seconds': metrics.get('total_seconds')}
    trainers = metrics.get('trainers')
    for item in trainers if isinstance(trainers, list) else [trainers]:
        # 只有 trainer rank 0 带 checkpoint_engine 指标，其余 rank 为 None。
        checkpoint = item.get('checkpoint_engine') if isinstance(item, dict) else None
        if isinstance(checkpoint, dict):
            info['changed_ratio'] = checkpoint.get('checkpoint_engine/changed_ratio')
            info['payload_mbytes'] = checkpoint.get('checkpoint_engine/payload_mbytes')
            info['send_seconds'] = item.get('send_seconds')
            break
    return info


def timed_sync(manager) -> tuple[float, dict]:
    started = perf_counter()
    manager.sync_weights()
    seconds = perf_counter() - started
    return seconds, delta_info(manager)


def build_batch(batch_size: int, seq_len: int) -> list[dict]:
    """构造已编码的最小训练 batch（InputFeature 列表）：全部 token 参与 loss，无需 tokenizer / 数据集。"""
    row = [(i % 1000) + 1 for i in range(seq_len)]
    return [{'input_ids': list(row), 'attention_mask': [1] * seq_len, 'labels': list(row)}
            for _ in range(batch_size)]


def resolve(result):
    """lazy_collect 打开时远程调用返回 callable，需要调用一次才会真正等待结果。"""
    return result() if callable(result) else result


def train_step(trainer, batch, max_grad_norm: float) -> float:
    """一次真实训练：前向 + 反向 + AdamW 更新（接口与 cookbook/transformers 一致）。返回耗时（秒）。

    结尾的 sync_device 用来等设备上的优化器 kernel 跑完，避免这部分时间漏进随后的 sync。
    """
    started = perf_counter()
    resolve(trainer.forward_backward(inputs=batch))
    resolve(trainer.clip_grad_and_step(max_grad_norm=max_grad_norm))
    resolve(trainer.sync_device())
    return perf_counter() - started


def run_benchmark(args) -> None:
    # 延迟导入：--compare 不需要 torch / Ray。
    from cookbook.checkpoint_engine.delta_validation import build_components, validation_stage
    from twinkle.checkpoint_engine import CheckpointEngineManager, DeltaSyncConfig

    log_progress(f'e2e step benchmark：backend={args.backend}, platform={args.platform}, model={args.model}, '
                 f'trainer={args.trainer_gpus}, rollout dp={args.dp} tp={args.tp}, '
                 f'batch={args.batch_size} seq={args.seq_len}')
    trainer, sampler = build_components(args)
    if args.backend == 'delta_sharded':
        manager = CheckpointEngineManager(
            trainer, sampler, platform=args.platform, mode='standalone', sync_backend='delta_sharded',
            delta_config=DeltaSyncConfig(**delta_config_kwargs(args)))
    else:
        manager = CheckpointEngineManager(trainer, sampler, platform=args.platform, mode='standalone')

    report = {'status': 'running', 'backend': args.backend, 'arguments': vars(args), 'seed': None, 'real_steps': []}

    def save_report():
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    try:
        # 必须先 set_optimizer 再做首次同步：同步会触发 FSDP 懒包装，包装时才会把优化器绑到分片参数上。
        trainer.set_optimizer('AdamW', lr=args.lr, weight_decay=args.weight_decay, foreach=False)
        batch = build_batch(args.batch_size, args.seq_len)

        # 首次 sync：delta 为 dense seed（建立快照和通信组），full 为首次全量；只记录，不进入统计。
        with validation_stage('首次 seed 同步（不计入统计）'):
            seconds, info = timed_sync(manager)
        report['seed'] = {'sync_seconds': seconds, **info}
        log_progress(f'[seed] sync={seconds:.3f}s {info}')
        save_report()

        # ---- real：训练计算 + 同步，变化比例由 AdamW 自然产生 ----
        total_real = args.warmup_steps + args.real_steps
        for index in range(total_real):
            warmup = index < args.warmup_steps
            train_seconds = train_step(trainer, batch, args.max_grad_norm)
            sync_seconds, info = timed_sync(manager)
            row = {'index': index, 'warmup': warmup, 'train_seconds': train_seconds, 'sync_seconds': sync_seconds,
                   'step_seconds': train_seconds + sync_seconds, **info}
            report['real_steps'].append(row)
            tag = 'warmup' if warmup else 'real'
            log_progress(f'[{tag} {index + 1}/{total_real}] train={train_seconds:.3f}s sync={sync_seconds:.3f}s '
                         f'step={row["step_seconds"]:.3f}s {info}')
            save_report()

        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=repr(error))
        log_progress(f'基准失败：{error!r}')
        raise
    finally:
        try:
            if args.backend == 'delta_sharded':
                with validation_stage('关闭增量同步会话，释放通信组和快照'):
                    manager.close_sync_session()
        finally:
            save_report()
            log_progress(f'报告已保存：{Path(args.report).resolve()}（status={report["status"]}）')

    print_single_summary(report)


def _med(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return median(values) if values else None


def _fmt(value, spec='.3f', suffix=''):
    return '-' if value is None else f'{value:{spec}}{suffix}'


def print_single_summary(report) -> None:
    measured = [row for row in report['real_steps'] if not row['warmup']]
    print(f"\nbackend={report['backend']}  real steps（中位数，{len(measured)} 步，已去掉 warmup）")
    print(f"  train={_fmt(_med(measured, 'train_seconds'))}s  sync={_fmt(_med(measured, 'sync_seconds'))}s  "
          f"step={_fmt(_med(measured, 'step_seconds'))}s  changed_ratio={_fmt(_med(measured, 'changed_ratio'), '.4f')}  "
          f"payload={_fmt(_med(measured, 'payload_mbytes'), '.1f', 'MB')}")


def compare_reports(paths, md_path=None) -> None:
    reports = {}
    for path in paths:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('backend') in reports:
            raise SystemExit(f'--compare 需要一份 delta_sharded 和一份 full 报告，但 {path} 的 backend 重复')
        if data.get('status') != 'passed':
            log_progress(f'警告：{path} 的 status={data.get("status")}，结果可能不完整')
        reports[data['backend']] = data
    if set(reports) != {'delta_sharded', 'full'}:
        raise SystemExit(f'--compare 需要 delta_sharded 与 full 各一份报告，实际: {sorted(reports)}')
    delta, full = reports['delta_sharded'], reports['full']

    lines = ['# 端到端训练 step 耗时：delta vs full', '']
    args = delta['arguments']
    lines += [f"环境：platform={args['platform']}，model={args['model']}，trainer={args['trainer_gpus']} 卡，"
              f"rollout dp={args['dp']} × tp={args['tp']}，batch={args['batch_size']} × seq={args['seq_len']}，"
              f"bucket={args['bucket_bytes'] >> 20}MB。",
              '一个 step = 训练计算（前向+反向+AdamW）+ 权重同步（阻塞到 rollout 就绪）。`speedup = full step / delta step`，'
              '>1 表示 delta 更快。', '']

    def real_rows(report):
        return [row for row in report['real_steps'] if not row['warmup']]

    d_real, f_real = real_rows(delta), real_rows(full)
    lines += ['## 真实 step（AdamW 自然产生的变化比例）', '',
              '| backend | train (s) | sync (s) | step (s) | changed_ratio | payload (MB) |',
              '| --- | --- | --- | --- | --- | --- |']
    for name, rows in (('delta_sharded', d_real), ('full', f_real)):
        lines.append(f"| {name} | {_fmt(_med(rows, 'train_seconds'))} | {_fmt(_med(rows, 'sync_seconds'))} | "
                     f"{_fmt(_med(rows, 'step_seconds'))} | {_fmt(_med(rows, 'changed_ratio'), '.4f')} | "
                     f"{_fmt(_med(rows, 'payload_mbytes'), '.1f')} |")
    d_step, f_step = _med(d_real, 'step_seconds'), _med(f_real, 'step_seconds')
    if d_step and f_step:
        lines += ['', f'speedup（real）= {f_step / d_step:.2f}x（full {f_step:.3f}s / delta {d_step:.3f}s）']

    text = '\n'.join(lines) + '\n'
    print(text)
    if md_path:
        Path(md_path).write_text(text, encoding='utf-8')
        log_progress(f'对比表已写入：{Path(md_path).resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--compare', nargs=2, metavar=('REPORT_A', 'REPORT_B'),
                        help='合并 delta_sharded 与 full 两份 JSON 报告并输出对比表（不启动 Ray）')
    parser.add_argument('--md', help='--compare 时把对比表另存为 markdown')
    add_platform_arguments(parser)
    parser.add_argument('--model', default='Qwen/Qwen3-0.6B', help='训练与 rollout 使用的同一模型')
    parser.add_argument('--backend', choices=('delta_sharded', 'full'), default='delta_sharded')
    parser.add_argument('--trainer-gpus', type=int, choices=(1, 2), default=2, help='FSDP2 训练卡数；1 表示不分片')
    parser.add_argument('--tp', type=int, choices=(1, 2), default=1, help='每个 rollout 副本的卡数')
    parser.add_argument('--dp', type=int, default=1, help='独立 rollout 副本数')
    parser.add_argument('--strategy', choices=('native_fsdp', 'accelerate'), default='native_fsdp')
    parser.add_argument('--cpu-offload', action='store_true', help='使用 Accelerate FSDP2 CPU offload')
    parser.add_argument('--gather-batch-size', type=int, default=32)
    parser.add_argument('--bucket-bytes', type=int, default=2048 << 20, help='默认 2048MB，与 verl 一致')
    parser.add_argument('--devices-per-node', type=int, help='每个节点的卡数，默认单节点运行')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='全局 batch size，按 DP 切分给 trainer 各卡，必须能被 --trainer-gpus 整除；默认等于 --trainer-gpus')
    parser.add_argument('--seq-len', type=int, default=128,
                        help='每条样本的 token 数。全参数 AdamW 的优化器状态很占显存（9B 模型每卡约 30~50GB），'
                             '默认取较小值，OOM 时调小或加 --strategy accelerate --cpu-offload')
    parser.add_argument('--lr', type=float, default=1e-5, help='AdamW 学习率；它决定 real step 自然产生的变化比例')
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--warmup-steps', type=int, default=1, help='real step 的预热步数（不进入统计）')
    parser.add_argument('--real-steps', type=int, default=3, help='参与统计的 real step 数')
    parser.add_argument('--report', default=None, help='JSON 报告路径，默认 e2e-<backend>.json')
    args = parser.parse_args()

    if args.compare:
        compare_reports(args.compare, args.md)
        return
    if args.report is None:
        args.report = f'e2e-{args.backend}.json'
    if args.dp < 1 or args.real_steps < 0 or args.warmup_steps < 0:
        parser.error('--dp 必须为正数，--real-steps/--warmup-steps 不能为负')
    if args.batch_size is None:
        args.batch_size = args.trainer_gpus
    if args.batch_size < 1 or args.seq_len < 1:
        parser.error('--batch-size 和 --seq-len 必须为正数')
    if args.batch_size % args.trainer_gpus:
        parser.error(f'--batch-size={args.batch_size} 必须能被 --trainer-gpus={args.trainer_gpus} 整除（forward_backward 按 DP 切分）')
    if args.cpu_offload and args.strategy != 'accelerate':
        parser.error('--cpu-offload 需要 --strategy accelerate')
    # 复用 delta_validation 的 worker extension 做预检（延迟导入常量所在模块会拉起 torch，所以直接写字符串）。
    run_preflight_or_exit(args, parser, 'cookbook.checkpoint_engine.delta_validation.ValidationWorker')
    run_benchmark(args)


if __name__ == '__main__':
    main()
