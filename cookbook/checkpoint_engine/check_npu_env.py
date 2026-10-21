# Copyright (c) ModelScope Contributors. All rights reserved.
"""检查当前环境是否满足 delta_sharded（HCCL/NPU 或 IPC/CUDA）增量权重同步的依赖。

只做导入和签名检查，不启动 Ray、不接触加速卡，可在任一节点单独运行：

    python -m cookbook.checkpoint_engine.check_npu_env            # 自动探测平台
    python -m cookbook.checkpoint_engine.check_npu_env --platform NPU

每项打印 [PASS]/[FAIL]/[WARN]，退出码 0 表示所有必需项通过。换机器或换镜像后先跑它，
可以在拉起几分钟的 vLLM 之前就发现 vLLM / vllm-ascend 版本不兼容的问题。
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect


class Checker:
    def __init__(self) -> None:
        self.failed = 0
        self.warned = 0

    def ok(self, name: str, detail: str = '') -> None:
        print(f'[PASS] {name}' + (f'：{detail}' if detail else ''))

    def fail(self, name: str, detail: str) -> None:
        self.failed += 1
        print(f'[FAIL] {name}：{detail}')

    def warn(self, name: str, detail: str) -> None:
        self.warned += 1
        print(f'[WARN] {name}：{detail}')

    def check(self, name: str, fn, *, required: bool = True) -> object:
        """运行 fn；返回 (detail_or_value)。约定：fn 返回字符串即 PASS 详情，抛异常即失败。"""
        try:
            detail = fn()
        except Exception as error:  # noqa: BLE001 - 汇总所有失败，不在第一项中止
            (self.fail if required else self.warn)(name, f'{type(error).__name__}: {error}')
            return None
        self.ok(name, detail if isinstance(detail, str) else '')
        return detail


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return 'not-installed'


def check_versions(checker: Checker) -> None:
    for package in ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'transformers'):
        version = _version(package.replace('_', '-'))
        print(f'       {package}: {version}')


def check_weight_transfer_engine(checker: Checker) -> None:
    def fn():
        from vllm.distributed.weight_transfer.base import WeightTransferEngine
        params = inspect.signature(WeightTransferEngine.__init__).parameters
        missing = {'config', 'vllm_config', 'device', 'model'} - set(params)
        if missing:
            raise RuntimeError(f'WeightTransferEngine.__init__ 缺少参数 {sorted(missing)}（需要 vLLM #44353）')
        return 'engine-owned WeightTransferEngine API 可用（vLLM #44353）'

    checker.check('WeightTransferEngine API (#44353)', fn)


def check_checkpoint_patch_api(checker: Checker) -> None:
    def fn():
        module = importlib.import_module('vllm.model_executor.model_loader.checkpoint_weight_patch')
        for attr in ('CheckpointWeightPatch', 'load_checkpoint_weight_patches'):
            if not hasattr(module, attr):
                raise RuntimeError(f'缺少 {attr}（需要 vLLM #50723）')
        sig = inspect.signature(module.load_checkpoint_weight_patches)
        if 'validate_unique_indices' not in sig.parameters:
            raise RuntimeError('load_checkpoint_weight_patches 缺少 validate_unique_indices 参数')
        return 'checkpoint patch API 可用（vLLM #50723）'

    checker.check('checkpoint patch API (#50723)', fn)


def check_layerwise_reload(checker: Checker) -> None:
    def fn():
        module = importlib.import_module('vllm.model_executor.model_loader.reload')
        for attr in ('initialize_layerwise_reload', 'finalize_layerwise_reload'):
            if not hasattr(module, attr):
                raise RuntimeError(f'缺少 {attr}（dense/full 同步依赖 layerwise reload）')
        return 'layerwise reload 生命周期可用'

    checker.check('layerwise reload API', fn)


def check_stateless_pg(checker: Checker) -> None:
    def fn():
        import dataclasses
        from vllm.distributed.utils import StatelessProcessGroup
        fields = {f.name for f in dataclasses.fields(StatelessProcessGroup)}
        # torch_utils.stateless_init_process_group 已兼容有/无 socket 两种版本，这里只提示。
        return f"StatelessProcessGroup 可导入（socket 字段：{'有' if 'socket' in fields else '无，走兼容分支'}）"

    checker.check('StatelessProcessGroup', fn)


def check_npu_hccl(checker: Checker) -> None:
    def torch_npu_available():
        import torch
        import torch_npu  # noqa: F401
        if not torch.npu.is_available():
            raise RuntimeError('torch.npu.is_available() 为 False（本节点看不到 NPU）')
        return f'torch.npu 可用，device_count={torch.npu.device_count()}'

    checker.check('torch_npu 设备', torch_npu_available)

    def pyhccl():
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator  # noqa: F401
        return 'vllm_ascend PyHcclCommunicator 可导入（delta HCCL broadcast 依赖）'

    checker.check('vllm-ascend PyHccl', pyhccl)

    def rl_config():
        # RL 场景必须支持 additional_config={'rl_config': {'enabled': True}}；仅检查配置类存在。
        from vllm_ascend import ascend_config  # noqa: F401
        return 'vllm_ascend.ascend_config 可导入（rl_config 开关所在模块）'

    checker.check('vllm-ascend rl_config 模块', rl_config, required=False)


def check_cuda_ipc(checker: Checker) -> None:
    def fn():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('torch.cuda.is_available() 为 False')
        return f'torch.cuda 可用，device_count={torch.cuda.device_count()}'

    checker.check('torch.cuda 设备', fn)


def check_twinkle_delta(checker: Checker) -> None:
    def fn():
        from twinkle.sampler.vllm_sampler.delta_weight_transfer import require_vllm_delta_support
        require_vllm_delta_support()
        return 'twinkle require_vllm_delta_support() 通过'

    checker.check('twinkle delta 依赖自检', fn)

    def refresh():
        from twinkle.sampler.vllm_sampler.weight_update_utils import refresh_derived_weights  # noqa: F401
        return 'refresh_derived_weights 可用（sparse 同步后刷新 Ascend 打包权重）'

    checker.check('twinkle refresh_derived_weights', refresh, required=False)


def detect_platform() -> str:
    try:
        import torch
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return 'NPU'
    except Exception:  # noqa: BLE001
        pass
    return 'GPU'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--platform', choices=('GPU', 'NPU'), help='默认自动探测（能导入 torch_npu 且 NPU 可用则 NPU）')
    args = parser.parse_args()
    platform = args.platform or detect_platform()

    print(f'== 依赖检查 platform={platform} ==')
    checker = Checker()
    check_versions(checker)
    print('-- vLLM 通用 delta 依赖 --')
    check_weight_transfer_engine(checker)
    check_checkpoint_patch_api(checker)
    check_layerwise_reload(checker)
    check_stateless_pg(checker)
    print(f'-- {platform} 专用依赖 --')
    if platform == 'NPU':
        check_npu_hccl(checker)
    else:
        check_cuda_ipc(checker)
    print('-- twinkle 侧 --')
    check_twinkle_delta(checker)

    print('== 汇总 ==')
    if checker.failed:
        print(f'{checker.failed} 项必需检查未通过，{checker.warned} 项告警。请先解决 [FAIL] 再运行 delta_validation。')
        return 1
    print(f'全部必需检查通过，{checker.warned} 项告警。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
