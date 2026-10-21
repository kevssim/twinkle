# Copyright (c) ModelScope Contributors. All rights reserved.
"""最小化的两机通信连通性测试，不依赖 twinkle / vLLM / Ray。

用途：当 delta_validation 跨机卡在 HCCL 建链时，用它判断是“网络本身不通”还是“HCCL 跨机不通”。
分两个后端测：
  - gloo：纯 TCP，验证两机三层网络 + MASTER_PORT 能否 rendezvous（不碰 NPU）。
  - hccl：昇腾集合通信，验证 NPU 跨机能否真正建链并做 all_reduce。

两个 Pod 各跑一条命令（先 gloo 再 hccl）。MASTER 必须是 rank 0 所在 Pod 的 IP。

  # Pod1（rank 0，172.61.8.99）
  python -m cookbook.checkpoint_engine.crossnode_comm_check \
      --backend gloo --rank 0 --world-size 2 --master-addr 172.61.8.99 --master-port 29555
  # Pod2（rank 1，172.61.10.44）
  python -m cookbook.checkpoint_engine.crossnode_comm_check \
      --backend gloo --rank 1 --world-size 2 --master-addr 172.61.8.99 --master-port 29555

gloo 通过后把 --backend 换成 hccl，--master-port 换一个新端口（如 29556）再各跑一次。
gloo 通、hccl 不通，基本可判定：该环境的 NPU 不具备跨机 HCCL（通常是缺 RDMA/RoCE）。
"""
from __future__ import annotations

import argparse
import datetime
import os
import socket


def log(message: str) -> None:
    print(f'[{datetime.datetime.now():%H:%M:%S}] [{socket.gethostname()}] {message}', flush=True)


def _run_stateless(args, torch, device) -> int:
    """用 delta 完全相同的 pyhccl 路径（stateless_init_process_group）建链并做 all_reduce。

    这正是 delta 超时的那条调用栈：
      stateless_init_process_group -> PyHcclCommunicator(pg, device) -> __init__ 里的 all_reduce。
    若此处也超时，说明问题在 pyhccl 初始化路径本身（与并发组/时间错位无关）；
    若此处 PASS，说明裸 pyhccl 单组可通，delta 的超时来自并发的第二个 HCCL 组或 join 时间错位。
    """
    from twinkle.utils import stateless_init_process_group

    log(f'开始 stateless_init_process_group（pyhccl）：rank={args.rank}/{args.world_size}, '
        f'master={args.master_addr}:{args.master_port}, device={device.index}')
    log('这条路径 = delta 的 build_delta_hccl_communicator；PyHcclCommunicator.__init__ 内部会做一次 all_reduce')
    try:
        comm = stateless_init_process_group(
            master_address=args.master_addr,
            master_port=args.master_port,
            rank=args.rank,
            world_size=args.world_size,
            device=device.index,
            backend='hccl',
        )
    except Exception as error:  # noqa: BLE001
        import traceback
        log(f'FAIL：stateless_init_process_group 抛异常：{type(error).__name__}: {error}')
        traceback.print_exc()
        return 1
    log('stateless_init_process_group 返回成功（communicator 已建立，内部 all_reduce 已通过）')

    try:
        tensor = torch.ones(1, device=device)
        # PyHcclCommunicator.all_reduce 是 out-of-place：结果在返回值里，不改入参。
        out = comm.all_reduce(tensor)
        torch.npu.synchronize()
        value = int(out.item())
        log(f'comm.all_reduce 完成，结果={value}（期望={args.world_size}）')
        if value != args.world_size:
            log('FAIL：all_reduce 结果不等于 world_size，通信异常')
            return 1
        log('PASS：pyhccl 跨机单组通信正常')
    except Exception as error:  # noqa: BLE001
        import traceback
        log(f'FAIL：comm.all_reduce 抛异常：{type(error).__name__}: {error}')
        traceback.print_exc()
        return 1
    return 0


def _build_pyhccl_group(args, torch, device, master_port, label):
    """用 delta 的 pyhccl 路径建一个组并做一次 all_reduce，返回 (comm, 是否正常)。"""
    from twinkle.utils import stateless_init_process_group

    log(f'[{label}] stateless_init_process_group：master_port={master_port}, '
        f"HCCL_IF_BASE_PORT={os.environ.get('HCCL_IF_BASE_PORT')}, "
        f"NPU_RANGE={os.environ.get('HCCL_NPU_SOCKET_PORT_RANGE')}")
    comm = stateless_init_process_group(
        master_address=args.master_addr, master_port=master_port,
        rank=args.rank, world_size=args.world_size, device=device.index, backend='hccl')
    out = comm.all_reduce(torch.ones(1, device=device))
    torch.npu.synchronize()
    value = int(out.item())
    ok = value == args.world_size
    log(f'[{label}] all_reduce 结果={value}（期望={args.world_size}）{"OK" if ok else "异常"}')
    return comm, ok


def _run_two_groups(args, torch, device) -> int:
    """在同一张卡上先后建两个并发 HCCL 组，复现 delta 的「第二个并发组」场景。

    delta 真实情况：trainer 卡上已有 FSDP 组、Pod2 卡上已有 vLLM TP 组，delta 组是第二个；
    且 trainer 侧 delta 组与 FSDP 组共用同一个 HCCL_IF_BASE_PORT。

    组 1 建好后不销毁（占住设备），再建组 2。
      - 默认：组 2 复用相同 HCCL 端口（= delta 与 FSDP 共享 base port 的情形）。
      - 指定 --second-base-port：组 2 换一套端口（验证「换端口即可解决」的修复方向）。
    """
    log('==== 建立组 1（占住设备，模拟已有的 FSDP / TP 通信域）====')
    try:
        comm1, ok1 = _build_pyhccl_group(args, torch, device, args.master_port, 'group1')
    except Exception as error:  # noqa: BLE001
        import traceback
        log(f'FAIL：组 1 建链失败：{type(error).__name__}: {error}')
        traceback.print_exc()
        return 1
    if not ok1:
        log('FAIL：组 1 本身就不正常，无法继续')
        return 1
    log('组 1 保持存活（不销毁），现在在同一张卡上建组 2')

    if args.second_base_port:
        base = args.second_base_port
        os.environ['HCCL_IF_BASE_PORT'] = str(base)
        os.environ['HCCL_HOST_SOCKET_PORT_RANGE'] = f'{base + 1000}-{base + 1511}'
        os.environ['HCCL_NPU_SOCKET_PORT_RANGE'] = f'{base + 2000}-{base + 2511}'
        log(f'==== 建立组 2（换一套 HCCL 端口 base={base}）====')
    else:
        log('==== 建立组 2（复用与组 1 相同的 HCCL 端口 = delta 与 FSDP 共享 base port 的情形）====')

    try:
        comm2, ok2 = _build_pyhccl_group(args, torch, device, args.master_port + 1, 'group2')
    except Exception as error:  # noqa: BLE001
        import traceback
        log(f'FAIL：组 2 建链失败（这就是 delta 的超时点）：{type(error).__name__}: {error}')
        traceback.print_exc()
        return 1
    if not ok2:
        log('FAIL：组 2 并发通信异常')
        return 1
    log('PASS：同一张卡上第二个并发 HCCL 组也正常')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backend', choices=('gloo', 'hccl'), required=True)
    parser.add_argument('--rank', type=int, required=True, help='本进程的全局 rank（rank 0 即 master）')
    parser.add_argument('--world-size', type=int, default=2)
    parser.add_argument('--master-addr', required=True, help='rank 0 所在 Pod 的 IP')
    parser.add_argument('--master-port', type=int, default=29555)
    parser.add_argument('--device', type=int, default=0, help='本机使用的本地 NPU 序号（hccl 后端用）')
    parser.add_argument('--timeout', type=int, default=120, help='rendezvous/集合通信超时秒数')
    parser.add_argument('--iface', help='指定网卡名，会设置 GLOO_SOCKET_IFNAME / HCCL_SOCKET_IFNAME')
    parser.add_argument(
        '--stateless', action='store_true',
        help='hccl 后端用 delta 完全相同的 vllm_ascend pyhccl 路径（stateless_init_process_group）建链，'
             '而非 torch.distributed。用于判断 delta 超时是否来自这条 pyhccl 初始化路径本身。')
    parser.add_argument(
        '--startup-delay', type=int, default=0,
        help='建链前本进程先睡眠的秒数。用于复现 delta 里 trainer（rank 0）很晚才 join 的时间错位，'
             '例如给 rank 0 设一个大的值，看非 0 rank 是否因等待超时。')
    parser.add_argument(
        '--two-groups', action='store_true',
        help='在同一张卡上先后建两个并发 HCCL 组（pyhccl 路径），复现 delta 的第二个并发组场景。')
    parser.add_argument(
        '--second-base-port', type=int, default=0,
        help='配合 --two-groups：给第二个组换一套 HCCL_IF_BASE_PORT（及派生端口段）。'
             '不指定则第二个组复用与第一个组相同的端口（= delta 与 FSDP 共享 base port）。')
    args = parser.parse_args()

    if args.iface:
        os.environ['GLOO_SOCKET_IFNAME'] = args.iface
        os.environ['HCCL_SOCKET_IFNAME'] = args.iface
        os.environ.setdefault('TP_SOCKET_IFNAME', args.iface)

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    os.environ['RANK'] = str(args.rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)

    import torch
    import torch.distributed as dist

    if args.backend == 'hccl':
        # 必须在创建进程组前导入并选卡，否则 HCCL 不知道用哪张 NPU。
        import torch_npu  # noqa: F401
        if not torch.npu.is_available():
            log('FAIL：torch.npu.is_available() 为 False，本机看不到 NPU')
            return 1
        torch.npu.set_device(args.device)
        device = torch.device('npu', args.device)
        log(f'使用 NPU device {args.device}，device_count={torch.npu.device_count()}')
    else:
        device = torch.device('cpu')

    if args.startup_delay > 0:
        import time
        log(f'建链前先睡眠 {args.startup_delay}s（模拟 join 时间错位）')
        time.sleep(args.startup_delay)

    if args.stateless:
        if args.backend != 'hccl':
            log('FAIL：--stateless 仅支持 hccl 后端')
            return 1
        return _run_stateless(args, torch, device)

    if args.two_groups:
        if args.backend != 'hccl':
            log('FAIL：--two-groups 仅支持 hccl 后端')
            return 1
        return _run_two_groups(args, torch, device)

    log(f'开始 init_process_group：backend={args.backend}, rank={args.rank}/{args.world_size}, '
        f'master={args.master_addr}:{args.master_port}, timeout={args.timeout}s')
    log('若进程在此长时间停住，说明两机无法在该 backend 上 rendezvous（gloo=TCP 不通；hccl=NPU 跨机建链失败）')
    try:
        dist.init_process_group(
            backend=args.backend, init_method='env://', rank=args.rank, world_size=args.world_size,
            timeout=datetime.timedelta(seconds=args.timeout))
    except Exception as error:  # noqa: BLE001
        log(f'FAIL：init_process_group 抛异常：{type(error).__name__}: {error}')
        return 1
    log('init_process_group 成功，进程组已建立')

    try:
        tensor = torch.ones(1, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if args.backend == 'hccl':
            torch.npu.synchronize()
        value = int(tensor.item())
        log(f'all_reduce 完成，结果={value}（期望={args.world_size}）')
        if value != args.world_size:
            log('FAIL：all_reduce 结果不等于 world_size，通信异常')
            return 1
        log('PASS：跨机通信正常')
    except Exception as error:  # noqa: BLE001
        log(f'FAIL：all_reduce 抛异常：{type(error).__name__}: {error}')
        return 1
    finally:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
